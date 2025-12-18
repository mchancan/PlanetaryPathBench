#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import math
import os
import shutil
import signal
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS_ROOT = os.path.join(ROOT, "results")
DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
PATHBENCH_SRC = os.path.join(ROOT, "PathBench", "src")
PATHBENCH_ROOT = os.path.dirname(PATHBENCH_SRC)
PLANNER_TIMEOUT_SECONDS = 60 # 1 minute per planner/map pair

if PATHBENCH_SRC not in sys.path:
	sys.path.insert(0, PATHBENCH_SRC)

import analyzer.analyzer as analyzer_module
from analyzer.analyzer import Analyzer
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.configuration import Configuration
from algorithms.configuration.maps.dense_map import DenseMap
from simulator.services.debug import DebugLevel
from simulator.services.services import Services
from algorithms.classic.graph_based.a_star import AStar
from algorithms.classic.graph_based.dijkstra import Dijkstra
from algorithms.classic.graph_based.potential_field import PotentialField
from algorithms.classic.sample_based.rrt import RRT
from algorithms.classic.sample_based.rrt_connect import RRT_Connect
from algorithms.classic.sample_based.sprm import SPRM
from algorithms.classic.testing.a_star_testing import AStarTesting
from algorithms.classic.testing.combined_online_lstm_testing import CombinedOnlineLSTMTesting
from algorithms.classic.testing.dijkstra_testing import DijkstraTesting
from algorithms.classic.testing.way_point_navigation_testing import WayPointNavigationTesting
from algorithms.lstm.a_star_waypoint import WayPointNavigation
from algorithms.lstm.combined_online_LSTM import CombinedOnlineLSTM
from algorithms.lstm.LSTM_tile_by_tile import OnlineLSTM

from adapters._common import auto_select_start_goal
from pathbench_wrappers import (
	WrappedAStar,
	WrappedRRTv4RRT,
	WrappedRRTv4Connect,
	WrappedRRTv4Dynamic,
	WrappedDijkstra,
	WrappedThetaStar,
	save_path_overlay_from_trace,
)

WRAPPED_ALGORITHMS: Sequence[Tuple[str, type]] = (
	("AStar", WrappedAStar),
	("RRTv4_RRT", WrappedRRTv4RRT),
	("RRTv4_RRTConnect", WrappedRRTv4Connect),
	("RRTv4_Dynamic", WrappedRRTv4Dynamic),
	("Dijkstra", WrappedDijkstra),
	("ThetaStar", WrappedThetaStar),
)

LSTM_VIEW_MODEL = "tile_by_tile_training_uniform_random_fill_10000_block_map_10000_house_10000_model"
LSTM_MAP_MODEL = "caelstm_section_lstm_training_uniform_random_fill_10000_block_map_10000_house_10000_model"

'''
Uncomment the planners below to enable PathBench planners like: A*, Dijkstra, RRT, RRT-Connect, WPN, LSTM modules, and LSTM Bagging.
These planners are disabled by default since they don't perform as well as the wrapped PathBench planners above
on the MarsPlanBench and MoonPlanBench datasets.
'''
NATIVE_PATHBENCH_ALGORITHMS: Sequence[Tuple[str, type, type, Tuple[List[Any], Dict[str, Any]]]] = (
	#("PB A*", AStar, AStarTesting, ([], {})),
	#("PB Dijkstra", Dijkstra, DijkstraTesting, ([], {})),
	#("PB RRT", RRT, BasicTesting, ([], {})),
	#("PB RRT-Connect", RRT_Connect, BasicTesting, ([], {})),
	#(
	#	"PB WPN",
	#	WayPointNavigation,
	#	WayPointNavigationTesting,
	#	(
	#		[],
	#		{
	#			"global_kernel_max_it": 10,
	#			"global_kernel": (CombinedOnlineLSTM, ([], {})),
	#		},
	#	),
	#),
	#("PB Map Module (CAE)", OnlineLSTM, BasicTesting, ([], {"load_name": LSTM_MAP_MODEL})),
	#("PB View Module (Online LSTM)", OnlineLSTM, BasicTesting, ([], {"load_name": LSTM_VIEW_MODEL})),
	#("LSTM Bagging", CombinedOnlineLSTM, CombinedOnlineLSTMTesting, ([], {})),
)

NATIVE_ALGORITHM_NAMES = {label for label, *_ in NATIVE_PATHBENCH_ALGORITHMS}

ALGO_TITLE_OVERRIDES = {
	"AStar": "AStar",
	"RRTv4_RRT": "RRT",
	"RRTv4_RRTConnect": "RRT Connect",
	"RRTv4_Dynamic": "Dynamic RRT",
	"Dijkstra": "Dijkstra",
	"ThetaStar": "ThetaStar",
	"LSTM Bagging": "LSTM Bagging",
	"PB A*": "A* (PathBench)",
	"PB Dijkstra": "Dijkstra (PathBench)",
	"PB RRT": "RRT (PathBench)",
	"PB RRT-Connect": "RRT Connect (PathBench)",
	"PB WPN": "WPN",
	"PB Map Module (CAE)": "CAE LSTM Module",
	"PB View Module (Online LSTM)": "View LSTM Module",
}

os.makedirs(RESULTS_ROOT, exist_ok=True)


class PlannerTimeout(RuntimeError):

	"""Raised when a planner exceeds the allowed planning time."""


@contextlib.contextmanager
def planner_time_limit(seconds: int):
	if seconds is None or seconds <= 0:
		yield
		return
	if not hasattr(signal, "setitimer"):
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


def _estimate_distance_to_goal(mp: DenseMap) -> float:
	start = getattr(mp, "start_xy", None)
	goal = getattr(mp, "goal_xy", None)
	if start is None or goal is None:
		return 0.0
	return float(math.hypot(goal[0] - start[0], goal[1] - start[1]))


def timeout_result(mp: DenseMap) -> Dict[str, Any]:
	dist = _estimate_distance_to_goal(mp)
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the strict PathBench analyzer over local occupancy maps",
	)
	parser.add_argument(
		"--data-dir",
		type=str,
		default=DEFAULT_DATA_DIR,
		help="Directory containing .npy map files (default: %(default)s)",
	)
	parser.add_argument(
		"--background-png-dir",
		type=str,
		default=None,
		help="Optional directory of PNG backdrops sharing names with the .npy maps",
	)
	parser.add_argument(
		"--moon",
		action="store_true",
		help="When set, use background images as-is (no rotation/flip).",
	)
	return parser.parse_args()


class Tee(io.TextIOBase):
	"""Mirror writes to multiple streams (console + file)."""

	def __init__(self, *streams: io.TextIOBase):
		super().__init__()
		self._streams = streams

	def write(self, data: str) -> int:  # type: ignore[override]
		for stream in self._streams:
			stream.write(data)
		return len(data)

	def flush(self) -> None:  # type: ignore[override]
		for stream in self._streams:
			stream.flush()


@contextlib.contextmanager
def working_directory(path: str):
	prev = os.getcwd()
	os.chdir(path)
	try:
		yield
	finally:
		os.chdir(prev)


def patch_matplotlib_show(output_dir: str) -> None:
	"""Redirect plt.show() to save all open figures into ``output_dir``."""

	output_dir = os.path.abspath(output_dir)

	def save_and_close_all(*_args, **_kwargs):
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		fig_numbers = list(plt.get_fignums())
		for idx, num in enumerate(fig_numbers):
			fig = plt.figure(num)
			# Post-process figure before saving: remove x/y labels and remap titles
			for ax in fig.get_axes():
				# remove axis labels and ticks
				try:
					ax.set_xlabel("")
					ax.set_ylabel("")
					ax.set_xticks([])
					ax.set_yticks([])
					ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
				except Exception:
					pass
				# reposition legend to lower-right corner when present
				try:
					handles, labels = ax.get_legend_handles_labels()
					if handles:
						ax.legend(handles, labels, loc="lower right")
				except Exception:
					pass
				# map titles to user-friendly planner names when possible
				title = ax.get_title() or fig.get_label() or ""
				mapped = None
				if title:
					if "AStar" in title or "A*" in title or title.strip().lower() == "astar":
						mapped = "A*"
					elif "PRMv2" in title or "PRM" in title:
						mapped = "PRM"
					elif "RRTv4" in title and "Connect" in title:
						mapped = "RRT Connect"
					elif "RRTv4" in title and ("Dynamic" in title or "Dynamic".lower() in title.lower()):
						mapped = "Dynamic RRT"
					elif "RRT" in title and "Connect" not in title:
						mapped = "RRT"
					elif "Dijkstra" in title:
						mapped = "Dijkstra"
					elif "Theta" in title:
						mapped = "Theta*"
					# fallback: try matching common substrings
					else:
						if "theta" in title.lower():
							mapped = "Theta*"
						elif "dijkstra" in title.lower():
							mapped = "Dijkstra"
						elif "rrt" in title.lower():
							# distinguish connect/dynamic by keyword
							if "connect" in title.lower():
								mapped = "RRT Connect"
							elif "dynamic" in title.lower():
								mapped = "Dynamic RRT"
							else:
								mapped = "RRT"
				if mapped:
					ax.set_title(mapped)
					# also set the figure label if empty
					try:
						fig.suptitle("")
					except Exception:
						pass
			outfile = os.path.join(output_dir, f"pathbench_plot_{timestamp}_{idx}.png")
			fig.savefig(outfile, bbox_inches="tight")
		if fig_numbers:
			print(f"Saved {len(fig_numbers)} analyzer plot(s) into {output_dir}")
		plt.close("all")

	plt.show = save_and_close_all  # type: ignore[assignment]

	def quiet_pause(*_args, **_kwargs):
		"""Ignore animation pauses to avoid excessive plot saves."""
		return None

	plt.pause = quiet_pause  # type: ignore[assignment]


def _style_categorical_axis(ax: Axes) -> None:
	ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)


def _normalize_algorithm_labels(frame: pd.DataFrame) -> pd.DataFrame:
	if "Algorithm" not in frame.columns:
		return frame
	frame = frame.copy()
	frame["Algorithm"] = frame["Algorithm"].astype(str).map(
		lambda name: ALGO_TITLE_OVERRIDES.get(name, name)
	)
	return frame


def save_metrics_figures(artifacts_dir: str) -> List[str]:
	summary_csv = os.path.join(artifacts_dir, "pbtest.csv")
	full_csv = os.path.join(artifacts_dir, "pbtestfull.csv")
	if not (os.path.exists(summary_csv) and os.path.exists(full_csv)):
		print(
			f"Skipping metrics plots: missing {'pbtest.csv' if not os.path.exists(summary_csv) else ''} {'pbtestfull.csv' if not os.path.exists(full_csv) else ''}"
		)
		return []

	df = _normalize_algorithm_labels(pd.read_csv(summary_csv))
	dffull = _normalize_algorithm_labels(pd.read_csv(full_csv))
	if df.empty or dffull.empty:
		print("Skipping metrics plots: analyzer CSVs are empty")
		return []

	sns.set_theme(style="whitegrid")
	metrics_dir = os.path.join(artifacts_dir, "metrics_plots")
	os.makedirs(metrics_dir, exist_ok=True)
	saved_paths: List[str] = []

	# Build a consistent color mapping for planners across all plots
	algorithm_sequence: List[str] = list(
		dict.fromkeys(list(df["Algorithm"].unique()) + list(dffull["Algorithm"].unique()))
	)
	if not algorithm_sequence:
		print("No algorithms found for metric plotting")
		return []
	pallette_source = sns.color_palette("tab20", n_colors=max(len(algorithm_sequence), 3))
	color_map = {alg: pallette_source[i % len(pallette_source)] for i, alg in enumerate(algorithm_sequence)}

	def palette_for_order(order: List[str]) -> List:
		return [color_map.get(alg, "grey") for alg in order]

	def filtered_order(frame: pd.DataFrame) -> List[str]:
		present = [alg for alg in algorithm_sequence if alg in frame["Algorithm"].unique()]
		return present if present else algorithm_sequence

	def _finalize(fig: plt.Figure, name: str) -> None:
		fig.tight_layout(pad=2.5, w_pad=1.5, h_pad=1.5)
		out_path = os.path.join(metrics_dir, name)
		fig.savefig(out_path, dpi=200)
		plt.close(fig)
		saved_paths.append(out_path)

	# Match PathBench's figure numbering order (plot4 -> plot1)
	fig1, axes1 = plt.subplots(ncols=3)
	fig1.set_size_inches(18, 12)
	bar_specs = [
		("Average Memory", "Average Memory vs. Algorithm ", "Average Memory (s)"),
		("Average Trajectory Smoothness", "Average Trajectory Smoothness vs. Algorithm ", "Average Trajectory Smoothness (rad/move)"),
		("Average Obstacle Clearance", "Average Obstacle Clearance vs. Algorithm ", "Average Obstacle Clearance"),
	]
	order_df = filtered_order(df)
	for ax, (y_col, title, ylabel) in zip(axes1, bar_specs):
		plot = sns.barplot(
			x="Algorithm",
			y=y_col,
			data=df,
			ax=ax,
			order=order_df,
			palette=palette_for_order(order_df),
		)
		_style_categorical_axis(plot)
		ax.set_title(title)
		ax.set_ylabel(ylabel)
	_finalize(fig1, "Figure_1.png")

	fig2, axes2 = plt.subplots(ncols=3)
	fig2.set_size_inches(18, 12)
	violin_specs_small = [
		("Memory", "Memory vs. Algorithm ", "Memory (KiB)"),
		("Trajectory Smoothness", "Trajectory Smoothness vs. Algorithm ", "Trajectory Smoothness (rad/move)"),
		("Obstacle Clearance", "Obstacle Clearance vs. Algorithm ", "Obstacle Clearance"),
	]
	order_full = filtered_order(dffull)
	for ax, (y_col, title, ylabel) in zip(axes2, violin_specs_small):
		plot = sns.violinplot(
			x="Algorithm",
			y=y_col,
			data=dffull,
			ax=ax,
			order=order_full,
			palette=palette_for_order(order_full),
		)
		_style_categorical_axis(plot)
		ax.set_title(title)
		ax.set_ylabel(ylabel)
	_finalize(fig2, "Figure_2.png")

	fig3, axes3 = plt.subplots(ncols=5)
	fig3.set_size_inches(18, 12)
	violin_specs_large = [
		("Time", "Time vs. Algorithm ", "Time (s)"),
		("Distance", "Distance vs. Algorithm ", "Distance"),
		("Distance from Goal", "Distance from Goal vs. Algorithm ", "Distance from Goal"),
		("Original Distance from Goal", "Original Distance from Goal vs. Algorithm ", "Original Distance from Goal"),
		("Path Deviation", "Path Deviation vs. Algorithm ", "Path Deviation"),
	]
	for ax, (y_col, title, ylabel) in zip(axes3, violin_specs_large):
		plot = sns.violinplot(
			x="Algorithm",
			y=y_col,
			data=dffull,
			ax=ax,
			order=order_full,
			palette=palette_for_order(order_full),
		)
		_style_categorical_axis(plot)
		ax.set_title(title)
		ax.set_ylabel(ylabel)
	_finalize(fig3, "Figure_3.png")

	fig4, axes4 = plt.subplots(ncols=5)
	fig4.set_size_inches(18, 11)
	bar_specs_large = [
		("Average Time", "Average Time vs. Algorithm ", "Average Time (s)"),
		("Average Distance", "Average Distance vs. Algorithm ", "Average Distance (m)"),
		("Average Steps", "Average Steps vs. Algorithm ", "Average Steps"),
		("Success Rate", "Success Rate vs. Algorithm ", "Success Rate"),
		("Average Path Deviation", "Average Path Deviation vs. Algorithm ", "Average Path Deviation"),
	]
	for ax, (y_col, title, ylabel) in zip(axes4, bar_specs_large):
		plot = sns.barplot(
			x="Algorithm",
			y=y_col,
			data=df,
			ax=ax,
			order=order_df,
			palette=palette_for_order(order_df),
		)
		_style_categorical_axis(plot)
		ax.set_title(title)
		ax.set_ylabel(ylabel)
	_finalize(fig4, "Figure_4.png")

	return saved_paths


def maybe_save_native_path_plot(result: Dict[str, Any], algo_label: str) -> None:
	if algo_label not in NATIVE_ALGORITHM_NAMES:
		return
	if not isinstance(result, dict):
		return
	map_obj = result.get("map")
	trace = result.get("trace")
	if map_obj is None or trace is None:
		return
	label = ALGO_TITLE_OVERRIDES.get(algo_label, algo_label)
	try:
		save_path_overlay_from_trace(map_obj, trace, label)
	except Exception as exc:
		map_name = getattr(map_obj, "name", "map")
		print(f"Warning: failed to save path plot for {label} on {map_name}: {exc}")


def build_dense_map(npy_path: str) -> DenseMap:
	grid = np.load(npy_path)
	if grid.ndim != 2:
		raise ValueError(f"Map {npy_path} must be a 2D array, got shape {grid.shape}")

	occ = (grid != 0).astype(int)
	start_xy, goal_xy = auto_select_start_goal(occ)
	start_xy_tuple = (int(start_xy[0]), int(start_xy[1]))
	goal_xy_tuple = (int(goal_xy[0]), int(goal_xy[1]))
	start_rc = (int(start_xy[1]), int(start_xy[0]))
	goal_rc = (int(goal_xy[1]), int(goal_xy[0]))

	dense = np.array(occ, dtype=int)
	dense[start_rc] = DenseMap.AGENT_ID
	dense[goal_rc] = DenseMap.GOAL_ID

	dm = DenseMap(dense, services=None)
	dm.name = os.path.basename(npy_path)
	dm.start_xy = start_xy_tuple  # type: ignore[attr-defined]
	dm.goal_xy = goal_xy_tuple  # type: ignore[attr-defined]
	dm.source_path = os.path.abspath(npy_path)  # type: ignore[attr-defined]
	return dm



def load_maps(data_dir: str) -> List[DenseMap]:
	npy_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
	if not npy_paths:
		raise FileNotFoundError(f"No .npy maps found under {data_dir}")
	return [build_dense_map(path) for path in npy_paths]


def write_map_summary(maps: Sequence[DenseMap], output_path: str) -> None:
	lines: List[str] = []
	missing_metadata = 0
	for mp in maps:
		name = getattr(mp, "name", os.path.basename(getattr(mp, "source_path", "unknown")))
		start = getattr(mp, "start_xy", None)
		goal = getattr(mp, "goal_xy", None)
		if start is None or goal is None:
			missing_metadata += 1
			continue
		lines.append(f"{name} start={tuple(start)} goal={tuple(goal)}")

	if not lines:
		print("Warning: No map metadata available to write summary file.")
		return

	with open(output_path, "w", encoding="utf-8") as fp:
		fp.write("\n".join(lines) + "\n")

	print(f"Saved map summary for {len(lines)} map(s) to {output_path}")
	if missing_metadata:
		print(f"Warning: skipped {missing_metadata} map(s) without start/goal metadata")


def prepare_configuration() -> Configuration:
	cfg = Configuration()
	cfg.algorithms.clear()
	cfg.analyzer = True
	cfg.num_dim = 2

	def register(label: str, alg_cls: type, testing_cls: Optional[type] = None, params: Optional[Tuple[List[Any], Dict[str, Any]]] = None) -> None:
		resolved_testing = testing_cls or getattr(alg_cls, "testing", BasicTesting)
		cfg.algorithms[label] = (alg_cls, resolved_testing, params or ([], {}))

	for label, alg_cls in WRAPPED_ALGORITHMS:
		register(label, alg_cls)

	for label, alg_cls, testing_cls, params in NATIVE_PATHBENCH_ALGORITHMS:
		register(label, alg_cls, testing_cls, params)

	return cfg


def reset_artifacts(artifacts_dir: str, path_plots_dir: str) -> None:
	if os.path.isdir(artifacts_dir):
		shutil.rmtree(artifacts_dir)
	os.makedirs(artifacts_dir, exist_ok=True)
	os.makedirs(path_plots_dir, exist_ok=True)


def run_strict_analyzer(
	analyzer: Analyzer,
	maps: Sequence[DenseMap],
	algorithms: Sequence[Tuple[type, type, Tuple[List[Any], Dict[str, Any]], str]],
) -> None:
	analyzer_module.algostring.clear()
	analysis_stream = io.StringIO()
	setattr(analyzer, "_Analyzer__analysis_stream", analysis_stream)
	services: Services = getattr(analyzer, "_Analyzer__services")

	services.debug.write("", timestamp=False, streams=[analysis_stream])
	services.debug.write(
		f"Starting basic analysis: number of maps = {len(maps)}, number of algorithms = {len(algorithms)}",
		end="\n\n",
		streams=[analysis_stream],
	)

	a_star_res = None
	algorithm_results: List[Dict[str, Any]] = [None] * len(algorithms)  # type: ignore[assignment]

	run_sim = getattr(analyzer, "_Analyzer__run_simulation")
	report_results = getattr(analyzer, "_Analyzer__report_results")
	tabulate_results = getattr(analyzer, "_Analyzer__tabulate_results")
	save_stream = getattr(analyzer, "_Analyzer__save_stream")

	for idx, (algorithm_type, testing_type, algo_params, display_name) in enumerate(algorithms):
		services.debug.write(f"{idx}. {display_name}", DebugLevel.BASIC, streams=[analysis_stream])
		results = []
		for map_idx, mp in enumerate(maps):
			timed_out = False
			try:
				with planner_time_limit(PLANNER_TIMEOUT_SECONDS):
					res = run_sim(mp, algorithm_type, testing_type, algo_params)
			except PlannerTimeout:
				timed_out = True
				res = timeout_result(mp)
				map_name = getattr(mp, "name", f"map_{map_idx}")
				services.debug.write(
					f"[Timeout] {display_name} exceeded {PLANNER_TIMEOUT_SECONDS}s on {map_name}; marking as failure",
					DebugLevel.BASIC,
					streams=[analysis_stream],
				)
			if timed_out:
				res["planner_timed_out"] = True
			else:
				res.pop("planner_timed_out", None)
				maybe_save_native_path_plot(res, display_name)
			results.append(res)
		baseline = results if a_star_res is None else a_star_res
		a_star_res, processed = report_results(results, baseline, algorithm_type, algorithms)
		algorithm_results[idx] = processed

	tabulate_results(algorithms, algorithm_results, with_indexing=True)
	save_stream("algo")
	setattr(analyzer, "_Analyzer__analysis_stream", None)


def copy_pathbench_log(timestamp: str, artifacts_dir: str) -> None:
	"""Copy PathBench's saved log into artifacts for convenience."""

	src = os.path.join(PATHBENCH_ROOT, "data", "algo_analysis_results_log_classical")
	if not os.path.exists(src):
		return

	dst = os.path.join(artifacts_dir, f"algo_analysis_results_log_classical_{timestamp}.txt")
	with open(src, "r") as in_f, open(dst, "w") as out_f:
		out_f.write(in_f.read())



def main() -> None:
	args = parse_args()
	data_dir = os.path.abspath(args.data_dir)
	if not os.path.isdir(data_dir):
		raise FileNotFoundError(f"Data directory not found: {data_dir}")
	if args.background_png_dir:
		background_dir = os.path.abspath(args.background_png_dir)
		if not os.path.isdir(background_dir):
			raise FileNotFoundError(f"Background PNG directory not found: {background_dir}")
		os.environ["PATHBENCH_BACKGROUND_PNG_DIR"] = background_dir
		# If --moon flag set, request original images (no rotate/flip)
		if getattr(args, "moon", False):
			os.environ["PATHBENCH_BACKGROUND_PNG_MOON"] = "1"

	timestamp = time.strftime("%Y%m%d_%H%M%S")
	artifacts_dir = os.path.join(RESULTS_ROOT, timestamp)
	path_plots_dir = os.path.join(artifacts_dir, "path_plots")
	os.environ["PATHBENCH_PATH_PLOTS_DIR"] = path_plots_dir
	os.environ["PATHBENCH_ARTIFACTS_DIR"] = artifacts_dir
	patch_matplotlib_show(path_plots_dir)
	reset_artifacts(artifacts_dir, path_plots_dir)
	maps = load_maps(data_dir)
	map_summary_path = os.path.join(artifacts_dir, "maps_start_goal.txt")
	write_map_summary(maps, map_summary_path)
	cfg = prepare_configuration()
	services = Services(cfg)
	analyzer = Analyzer(services)

	log_path = os.path.join(artifacts_dir, f"pathbench_analysis_{timestamp}.txt")

	algorithms = [(*cfg.algorithms[name], name) for name in cfg.algorithms]

	with open(log_path, "w") as log_file:
		tee = Tee(sys.stdout, log_file)
		with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
			with working_directory(artifacts_dir):
				run_strict_analyzer(analyzer, maps, algorithms)
			copy_pathbench_log(timestamp, artifacts_dir)

	saved_figures = save_metrics_figures(artifacts_dir)

	print(f"Analysis finished. Artifacts saved under {artifacts_dir}")
	print(f"- Aggregates: {os.path.join(artifacts_dir, 'pbtest.csv')}")
	print(f"- Full results: {os.path.join(artifacts_dir, 'pbtestfull.csv')}")
	print(f"- Log: {log_path}")
	print(f"- Map summary: {map_summary_path}")
	if saved_figures:
		print("- Metric figures:")
		for path in saved_figures:
			print(f"  • {path}")


if __name__ == "__main__":
	main()
