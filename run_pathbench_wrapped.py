#!/usr/bin/env python3
"""Run PathBench analyzers over local .npy maps using the adapter wrappers.

This utility mirrors PathBench's analyzer workflow but sources maps from the
``data/`` directory and reuses the PythonRobotics adapters already wired into
``run_planner.py`` / ``run_individual_planner.py``. It emits the familiar PathBench
CSVs plus a textual log inside ``artifacts/``.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import math
import os
import signal
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(__file__)
PATHBENCH_SRC = os.path.join(ROOT, "PathBench", "src")
if PATHBENCH_SRC not in sys.path:
    sys.path.insert(0, PATHBENCH_SRC)

from algorithms.basic_testing import BasicTesting
from algorithms.configuration.configuration import Configuration
from algorithms.configuration.maps.dense_map import DenseMap
from analyzer.analyzer import Analyzer
from simulator.services.services import Services

from pathbench_wrappers import (
    WrappedAStar,
    WrappedPRM,
    WrappedPRMv2,
    WrappedRRTv4RRT,
    WrappedRRTv4Connect,
    WrappedRRTv4Dynamic,
    WrappedDijkstra,
    WrappedThetaStar,
    WrappedPotentialField,
)

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
PLANNER_TIMEOUT_SECONDS = 300


class PlannerTimeout(RuntimeError):
    """Raised when a planner exceeds the wall-clock time limit."""


@contextlib.contextmanager
def planner_time_limit(seconds: int):
    if seconds is None or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handle_timeout(signum, frame):  # noqa: ARG001 - required signature
        raise PlannerTimeout(f"Planner exceeded {seconds} seconds")

    prev_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev_handler)


def _estimate_distance(dm: DenseMap) -> float:
    start = getattr(dm, "start_xy", None)
    goal = getattr(dm, "goal_xy", None)
    if start is None or goal is None:
        return 0.0
    return float(math.hypot(goal[0] - start[0], goal[1] - start[1]))


def timeout_result(dm: DenseMap) -> Dict:
    dist = _estimate_distance(dm)
    return {
        "goal_found": False,
        "total_steps": 0,
        "total_distance": 0.0,
        "smoothness_of_trajectory": 0.0,
        "obstacle_clearance": 0.0,
        "total_time": float(PLANNER_TIMEOUT_SECONDS),
        "distance_to_goal": dist,
        "original_distance_to_goal": dist,
        "memory": 0.0,
        "planner_timed_out": True,
    }

AlgorithmEntry = Tuple[type, type, Tuple[Sequence, Dict], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PathBench adapters on local maps and collect metrics",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(ROOT, "data"),
        help="Directory containing .npy map files (default: %(default)s)",
    )
    return parser.parse_args()


def build_map_from_npy(npy_path: str) -> DenseMap:
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"Map {npy_path} must be 2D, got shape {arr.shape}")

    occ = (arr != 0).astype(int)
    free_idx = np.argwhere(occ == 0)
    if free_idx.size == 0:
        start = (0, 0)
        goal = (0, 0)
    else:
        start = tuple(map(int, free_idx[0]))
        goal = tuple(map(int, free_idx[-1]))

    grid = np.array(occ, dtype=int)
    grid[start] = DenseMap.AGENT_ID
    grid[goal] = DenseMap.GOAL_ID

    dm = DenseMap(grid, services=None)
    dm.name = os.path.basename(npy_path)
    dm.start_xy = (int(start[1]), int(start[0]))  # type: ignore[attr-defined]
    dm.goal_xy = (int(goal[1]), int(goal[0]))  # type: ignore[attr-defined]
    return dm


def run_algorithms(maps: List[DenseMap]) -> Dict[str, List[Dict]]:
    cfg = Configuration()
    services = Services(cfg)
    analyzer = Analyzer(services)

    algorithms: Sequence[AlgorithmEntry] = [
        (WrappedAStar, BasicTesting, ([], {}), "AStar"),
        (WrappedPRM, BasicTesting, ([], {}), "PRM"),
        (WrappedPRMv2, BasicTesting, ([], {}), "PRMv2"),
        (WrappedRRTv4RRT, BasicTesting, ([], {}), "RRTv4_RRT"),
        (WrappedRRTv4Connect, BasicTesting, ([], {}), "RRTv4_RRTConnect"),
        (WrappedRRTv4Dynamic, BasicTesting, ([], {}), "RRTv4_Dynamic"),
        (WrappedDijkstra, BasicTesting, ([], {}), "Dijkstra"),
        (WrappedThetaStar, BasicTesting, ([], {}), "ThetaStar"),
        (WrappedPotentialField, BasicTesting, ([], {}), "PotentialField"),
    ]

    results: Dict[str, List[Dict]] = {}
    for idx, (alg_cls, testing_cls, params, name) in enumerate(algorithms):
        print(f"Running algorithm {idx + 1}/{len(algorithms)}: {name}")
        per_map: List[Dict] = []
        for dm in maps:
            try:
                with planner_time_limit(PLANNER_TIMEOUT_SECONDS):
                    res = analyzer._Analyzer__run_simulation(dm, alg_cls, testing_cls, params)
            except PlannerTimeout:
                map_name = getattr(dm, "name", os.path.basename(getattr(dm, "source_path", "map")))
                print(
                    f"[Timeout] {name} exceeded {PLANNER_TIMEOUT_SECONDS}s on {map_name}; marking as failure"
                )
                res = timeout_result(dm)
            except Exception as exc:  # pragma: no cover - PathBench runtime errors
                print(f"Simulation error for {name} on map {dm.name}: {exc}")
                res = {
                    "goal_found": False,
                    "total_steps": 0,
                    "total_distance": 0,
                    "smoothness_of_trajectory": 0,
                    "obstacle_clearance": 0,
                    "total_time": 0,
                    "distance_to_goal": 0,
                    "original_distance_to_goal": 0,
                    "memory": 0,
                }
            per_map.append(res)
        results[name] = per_map
    return results


def write_outputs(results: Dict[str, List[Dict]], timestamp: str) -> None:
    pbtest_path = os.path.join(ARTIFACTS_DIR, "pbtest.csv")
    pbtestfull_path = os.path.join(ARTIFACTS_DIR, "pbtestfull.csv")
    log_path = os.path.join(ARTIFACTS_DIR, f"pathbench_analysis_{timestamp}.txt")

    with open(pbtest_path, "w", newline="") as fh:
        fieldnames = [
            "Algorithm",
            "Average Path Deviation",
            "Success Rate",
            "Average Time",
            "Average Steps",
            "Average Distance",
            "Average Distance from Goal",
            "Average Original Distance from Goal",
            "Average Trajectory Smoothness",
            "Average Obstacle Clearance",
            "Average Search Space",
            "Average Memory",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name, res_list in results.items():
            n = len(res_list)
            if n == 0:
                continue
            success_rate = round(sum(1 for r in res_list if r.get("goal_found")) / n * 100, 2)
            avg_steps = round(sum(r.get("total_steps", 0) for r in res_list) / n, 2)
            avg_distance = round(sum(r.get("total_distance", 0) for r in res_list) / n, 2)
            avg_time = round(sum(r.get("total_time", 0) for r in res_list) / n, 4)
            avg_dist_goal = round(sum(r.get("distance_to_goal", 0) for r in res_list) / n, 2)
            avg_orig_dist = round(sum(r.get("original_distance_to_goal", 0) for r in res_list) / n, 2)
            avg_smooth = round(sum(r.get("smoothness_of_trajectory", 0) for r in res_list) / n, 2)
            avg_clear = round(sum(r.get("obstacle_clearance", 0) for r in res_list) / n, 2)
            avg_mem = round(sum(r.get("memory", 0) for r in res_list) / n, 2)
            writer.writerow(
                {
                    "Algorithm": name,
                    "Average Path Deviation": 0,
                    "Success Rate": success_rate,
                    "Average Time": avg_time,
                    "Average Steps": avg_steps,
                    "Average Distance": avg_distance,
                    "Average Distance from Goal": avg_dist_goal,
                    "Average Original Distance from Goal": avg_orig_dist,
                    "Average Trajectory Smoothness": avg_smooth,
                    "Average Obstacle Clearance": avg_clear,
                    "Average Search Space": 0,
                    "Average Memory": avg_mem,
                }
            )

    with open(pbtestfull_path, "w", newline="") as fh:
        fieldnames = [
            "Algorithm",
            "Time",
            "Distance",
            "Distance from Goal",
            "Path Deviation",
            "Original Distance from Goal",
            "Trajectory Smoothness",
            "Obstacle Clearance",
            "Search Space",
            "Memory",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name, res_list in results.items():
            for r in res_list:
                writer.writerow(
                    {
                        "Algorithm": name,
                        "Time": r.get("total_time", 0),
                        "Distance": r.get("total_distance", 0),
                        "Distance from Goal": r.get("distance_to_goal", 0),
                        "Path Deviation": 0,
                        "Original Distance from Goal": r.get("original_distance_to_goal", 0),
                        "Trajectory Smoothness": r.get("smoothness_of_trajectory", 0),
                        "Obstacle Clearance": r.get("obstacle_clearance", 0),
                        "Search Space": 0,
                        "Memory": r.get("memory", 0),
                    }
                )

    with open(log_path, "w") as fh:
        for name, res_list in results.items():
            fh.write(f"Algorithm: {name}\n")
            for idx, r in enumerate(res_list):
                fh.write(
                    f"  Map {idx}: goal_found={r.get('goal_found')}, time={r.get('total_time')}, "
                    f"steps={r.get('total_steps')}, distance={r.get('total_distance')}\n"
                )
            fh.write("\n")

    print(f"Analysis finished. CSVs/logs saved under {ARTIFACTS_DIR} (timestamp {timestamp}).")



def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    npy_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if not npy_paths:
        print("No .npy maps found in data/")
        sys.exit(1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path_plots_dir = os.path.join(ARTIFACTS_DIR, f"path_plots_{timestamp}")
    os.environ["PATHBENCH_PATH_PLOTS_DIR"] = path_plots_dir
    os.makedirs(path_plots_dir, exist_ok=True)

    maps = [build_map_from_npy(path) for path in npy_paths]
    results = run_algorithms(maps)
    write_outputs(results, timestamp)


if __name__ == "__main__":  # pragma: no cover
    main()
