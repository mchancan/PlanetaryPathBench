"""Wrappers that adapt run_planner adapters into PathBench Algorithm classes.

Each generated class subclasses PathBench's Algorithm API and, when run by
PathBench's Simulator, will call our existing adapters (from `adapters/`) to
compute a path on the DenseMap converted from an occupancy .npy map.

The wrapper then replays the produced path by calling Algorithm.move_agent()
so PathBench's testing/analysis code can compute metrics in the usual way.
"""
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as _plt
from matplotlib import colors as _colors
from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.maps.dense_map import DenseMap
from algorithms.configuration.entities.trace import Trace
from structures import Point
import numpy as _np
def _grid_line_sequence(frm: Point, to: Point):
    x0, y0 = int(frm.x), int(frm.y)
    x1, y1 = int(to.x), int(to.y)
    points = [Point(x0, y0)]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x_step = 1 if x0 < x1 else -1
    y_step = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while x != x1 or y != y1:
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += x_step
        if e2 < dx:
            err += dx
            y += y_step
        points.append(Point(x, y))
    return points



def _safe_filename_component(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _dense_map_to_display(dm: DenseMap) -> _np.ndarray:
    grid = _np.transpose(_np.array(dm.grid, copy=False))
    display = _np.zeros_like(grid, dtype=int)
    display[grid == DenseMap.WALL_ID] = 1
    display[grid == DenseMap.EXTENDED_WALL_ID] = 2
    display[grid == DenseMap.AGENT_ID] = 3
    display[grid == DenseMap.GOAL_ID] = 4
    return display


_TITLE_OVERRIDES = {
    "AStar": "AStar",
    "PRM": "PRM",
    "PRMv2": "PRM",
    "RRTv4_RRT": "RRT",
    "RRTv4_RRTConnect": "RRT Connect",
    "RRTv4_Dynamic": "Dynamic RRT",
    "Dijkstra": "Dijkstra",
    "ThetaStar": "ThetaStar",
}

_BACKGROUND_CACHE: Dict[Tuple[str, str], Optional[_np.ndarray]] = {}
_PNG_INDEX_CACHE: Dict[str, Dict[str, str]] = {}
_PNG_SUFFIX_REPLACEMENTS = (
    "_occupancy_map",
    "_occupancy",
    "_occ",
    "_map",
)
_PNG_SUFFIX_ENRICHMENTS = ("_overlay_rgb", "_overlay", "_rgb")

# Supported background image extensions (lowercase)
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")

def _normalized_keys_for_stem(stem: str) -> Tuple[str, ...]:
    root = stem.lower().strip()
    if not root:
        return tuple()
    keys = {root}
    for suffix in _PNG_SUFFIX_ENRICHMENTS:
        if root.endswith(suffix):
            trimmed = root[: -len(suffix)]
            if trimmed:
                keys.add(trimmed)
    return tuple(keys)


def _build_png_index(png_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    try:
        entries = os.listdir(png_dir)
    except FileNotFoundError:
        return index
    for entry in entries:
        if not entry.lower().endswith(_IMAGE_EXTS):
            continue
        stem = os.path.splitext(entry)[0]
        for key in _normalized_keys_for_stem(stem):
            index.setdefault(key, os.path.join(png_dir, entry))
    return index


def _candidate_background_keys(base_name: str) -> Tuple[str, ...]:
    base = base_name.lower()
    keys = [base]
    for suffix in _PNG_SUFFIX_REPLACEMENTS:
        if base.endswith(suffix):
            trimmed = base[: -len(suffix)]
            if trimmed:
                keys.append(trimmed)
                for enrich in _PNG_SUFFIX_ENRICHMENTS:
                    keys.append(f"{trimmed}{enrich}")
            break
    # preserve order while removing duplicates
    seen = set()
    ordered: List[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return tuple(ordered)


def _map_base_name(dm: DenseMap) -> str:
    source_path = getattr(dm, "source_path", None)
    base = os.path.splitext(os.path.basename(source_path or (dm.name or "map")))[0]
    if not base:
        return "map"
    return base


def _resolve_background_png(dm: DenseMap) -> Optional[_np.ndarray]:
    png_dir = os.environ.get("PATHBENCH_BACKGROUND_PNG_DIR")
    if not png_dir:
        return None
    base_name = _map_base_name(dm)
    cache_key = (png_dir, base_name.lower())
    if cache_key in _BACKGROUND_CACHE:
        return _BACKGROUND_CACHE[cache_key]
    index = _PNG_INDEX_CACHE.get(png_dir)
    if index is None:
        index = _build_png_index(png_dir)
        _PNG_INDEX_CACHE[png_dir] = index
    candidate = None
    for key in _candidate_background_keys(base_name):
        candidate = index.get(key)
        if candidate:
            break
    if candidate is None:
        # try a fresh index in case new PNGs were added mid-run
        index = _build_png_index(png_dir)
        _PNG_INDEX_CACHE[png_dir] = index
        for key in _candidate_background_keys(base_name):
            candidate = index.get(key)
            if candidate:
                break
    image = None
    if candidate and os.path.exists(candidate):
        try:
            data = _plt.imread(candidate)
            if data.ndim == 2:
                data = _np.stack([data] * 3, axis=-1)
            if data.ndim == 3 and data.shape[2] > 4:
                data = data[:, :, :4]
            # Allow callers to request the original image be used (no rotate/flip)
            _use_original = os.environ.get("PATHBENCH_BACKGROUND_PNG_MOON")
            if _use_original and _use_original.lower() in ("1", "true", "yes", "y"):
                image = data
            else:
                rotated = _np.rot90(data, -1)
                image = _np.fliplr(rotated)
        except Exception:
            image = None
    _BACKGROUND_CACHE[cache_key] = image
    return image


def _clean_algo_label(label: str) -> str:
    if not label:
        return "Path"
    return _TITLE_OVERRIDES.get(label, label)


def _save_path_overlay(dm: DenseMap, rx, ry, algo_label: str, start_xy=None, goal_xy=None) -> None:
    if rx is None or ry is None or len(rx) == 0 or len(rx) != len(ry):
        return
    out_dir = os.environ.get("PATHBENCH_PATH_PLOTS_DIR")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    display = _dense_map_to_display(dm)
    background = _resolve_background_png(dm)
    fig, ax = _plt.subplots(figsize=(6, 6))

    # Build a warm neutral gradient for occupancy categories, then append agent/goal colors
    gradient_steps = 3
    vals = _np.ones((gradient_steps, 4))
    vals[:, 0] = _np.linspace(245 / 256, 1, gradient_steps)
    vals[:, 1] = _np.linspace(222 / 256, 1, gradient_steps)
    vals[:, 2] = _np.linspace(179 / 256, 1, gradient_steps)
    neutral_colors = [_colors.to_hex(row[:3]) for row in vals]
    #"#ffffff", "#1f1f1f", "#555555", "#1f77b4", "#2ca02c"]
    cmap_colors = [
        "#ffffff",  # free space
        neutral_colors[1],  # obstacles
        "#555555",  # extended obstacles
        "#1f77b4",        # agent
        "#2ca02c",        # goal
    ]
    cmap = _colors.ListedColormap(cmap_colors)
    norm = _colors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    extent = (0, display.shape[1], 0, display.shape[0])
    if background is not None:
        ax.imshow(background, origin="lower", extent=extent, interpolation="bilinear")
    else:
        ax.imshow(
            display,
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="nearest",
            extent=extent,
        )
    agent_pos = None
    goal_pos = None
    if start_xy is not None:
        agent_pos = start_xy
    else:
        agent = getattr(dm, "agent", None)
        if agent is not None:
            agent_pos = (agent.position.x, agent.position.y)
    if goal_xy is not None:
        goal_pos = goal_xy
    else:
        goal = getattr(dm, "goal", None)
        if goal is not None:
            goal_pos = (goal.position.x, goal.position.y)
    if agent_pos is not None:
        ax.scatter(agent_pos[0], agent_pos[1], marker="s", s=36, c="#1f77b4", label="start")
    if goal_pos is not None:
        ax.scatter(goal_pos[0], goal_pos[1], marker="*", s=60, c="#2ca02c", label="goal")
    #ax.plot(rx, ry, color="#d62728", linewidth=2, label="path")
    #ax.plot(rx, ry, color="#E0FFFF", linewidth=2, label="path")
    ax.plot(rx, ry, color="#FF2400", linewidth=1, label="path")
    ax.set_title(_clean_algo_label(algo_label))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize="small")
    _safe_algo = _safe_filename_component(algo_label)
    base_name = dm.name or "map"
    _safe_map = _safe_filename_component(os.path.splitext(base_name)[0])
    out_path = os.path.join(out_dir, f"{_safe_map}_algo-{_safe_algo}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    _plt.close(fig)


def _quantize_path(rx, ry):
    points = []
    for x, y in zip(rx, ry):
        try:
            xi = int(math.floor(float(x)))
            yi = int(math.floor(float(y)))
            pt = Point(xi, yi)
        except Exception:
            return []
        if not points or points[-1] != pt:
            points.append(pt)
    return points


def _expand_path(points, map_obj: DenseMap):
    if not points:
        return []
    expanded = [points[0]]
    for current, nxt in zip(points, points[1:]):
        if current == nxt:
            continue
        try:
            segment = _grid_line_sequence(current, nxt)
        except Exception:
            return []
        if not segment:
            continue
        try:
            if not all(map_obj.is_agent_valid_pos(pt) for pt in segment):
                return []
        except Exception:
            return []
        if expanded[-1] == segment[0]:
            expanded.extend(segment[1:])
        else:
            expanded.extend(segment)
    return expanded


def _replay_path(map_obj: DenseMap, rx, ry, move_agent) -> bool:
    """Move the PathBench agent along the adapter path, step-by-step."""
    if rx is None or ry is None:
        return False
    try:
        total = len(rx)
    except TypeError:
        return False
    if total == 0 or total != len(ry):
        return False

    quantized = _quantize_path(rx, ry)
    start = map_obj.agent.position
    if not quantized:
        return False
    if quantized[0] != start:
        quantized.insert(0, Point(start.x, start.y))

    expanded = _expand_path(quantized, map_obj)
    if not expanded:
        return False
    if expanded[0] != start:
        expanded = [start] + expanded

    for step in expanded[1:]:
        try:
            result = move_agent(step)
        except Exception:
            return False
        if result is False:
            return False

    try:
        return map_obj.is_goal_reached(map_obj.agent.position)
    except Exception:
        return False


def _orient_path(rx, ry, start, goal):
    if not rx or not ry or len(rx) != len(ry):
        return rx, ry
    head = (rx[0], ry[0])
    tail = (rx[-1], ry[-1])
    start = tuple(start)
    goal = tuple(goal)
    def score(path_start, path_end):
        return math.hypot(path_start[0] - start[0], path_start[1] - start[1]) + \
            math.hypot(path_end[0] - goal[0], path_end[1] - goal[1])
    forward = score(head, tail)
    reverse = score(tail, head)
    if reverse + 1e-6 < forward:
        return list(reversed(rx)), list(reversed(ry))
    return rx, ry


def make_adapter_algorithm(adapter_module_name: str, display_name: str = None):
    """Return a new Algorithm subclass that wraps adapters.<adapter_module_name>.run

    adapter_module_name: module name under `adapters`, e.g. 'a_star' or 'rrtv4_rrt'
    """
    if display_name is None:
        display_name = adapter_module_name

    class AdapterAlgorithm(Algorithm):
        testing = BasicTesting

        def __init__(self, services, testing: BasicTesting = None):
            super().__init__(services, testing)

        def set_display_info(self):
            # No custom displays required for these wrappers
            return []

        def _find_path_internal(self) -> None:
            # Map available through self._get_grid()
            m = self._get_grid()

            # Extract occupancy array expected by our adapters: 1 for obstacle, 0 for free
            # DenseMap stores integer IDs; WALL_ID == 1, CLEAR_ID == 0. Build occupancy as 0/1 array.
            try:
                grid = _np.array(m.grid, copy=True)
            except Exception:
                # Fallback: try accessing via .grid attribute
                grid = _np.array(getattr(m, 'grid'))

            # DenseMap stores a transposed grid internally by default; transpose back so the
            # adapters receive the same orientation as the original occupancy npy files.
            grid = _np.transpose(grid)

            # Create occupancy: 1 where grid equals WALL_ID (DenseMap.WALL_ID == 1)
            occ = (grid == 1).astype(int)

            # Get start and goal from DenseMap agent/goal positions
            agent_pt = m.agent.position
            goal_pt = m.goal.position

            sx, sy = int(agent_pt.x), int(agent_pt.y)
            gx, gy = int(goal_pt.x), int(goal_pt.y)

            # Call the adapter
            try:
                mod = __import__('adapters.' + adapter_module_name, fromlist=['*'])
                run_fn = getattr(mod, 'run')
            except Exception:
                # Try package-qualified import
                mod = __import__('PathPlanning.adapters.' + adapter_module_name, fromlist=['*'])
                run_fn = getattr(mod, 'run')

            rx, ry = run_fn(occ, [sx, sy], [gx, gy])
            rx, ry = _orient_path(rx, ry, (sx, sy), (gx, gy))
            try:
                _save_path_overlay(m, rx, ry, display_name)
            except Exception:
                pass

            # Replay the path through the PathBench map by moving the agent step-by-step
            # The adapters generally return lists of floats that correspond to map grid coords
            if rx is not None and ry is not None:
                try:
                    path_len = len(rx)
                    match_len = len(ry)
                except TypeError:
                    path_len = match_len = 0
                if path_len > 0 and path_len == match_len:
                    print(f"[PathBench Wrapper] {display_name} path length {path_len}")
                    success = _replay_path(m, rx, ry, self.move_agent)
                    if not success:
                        print(f"[PathBench Wrapper] Replay terminated early for {display_name}; PathBench may mark this run as failure.")

    AdapterAlgorithm.__name__ = f"Wrapped_{display_name}"
    return AdapterAlgorithm


# Create named classes for the planners we intend to benchmark
WrappedAStar = make_adapter_algorithm('a_star', 'AStar')
WrappedPRM = make_adapter_algorithm('prm', 'PRM')
WrappedPRMv2 = make_adapter_algorithm('prm_occ', 'PRMv2')
WrappedRRTv4RRT = make_adapter_algorithm('rrtv4_rrt', 'RRTv4_RRT')
WrappedRRTv4Connect = make_adapter_algorithm('rrtv4_connect', 'RRTv4_RRTConnect')
WrappedRRTv4Dynamic = make_adapter_algorithm('rrtv4_dynamic', 'RRTv4_Dynamic')
WrappedDijkstra = make_adapter_algorithm('dijkstra', 'Dijkstra')
WrappedThetaStar = make_adapter_algorithm('thetastar', 'ThetaStar')
WrappedPotentialField = make_adapter_algorithm('potential_field', 'PotentialField')

def save_path_overlay_from_trace(dm: DenseMap, trace: Sequence[Trace], algo_label: str) -> None:
    """Render a path overlay PNG from a DenseMap trace for native PathBench planners."""
    if dm is None or trace is None:
        return
    points = []
    for entry in trace:
        pos = getattr(entry, "position", None)
        if pos is None:
            continue
        x_val = getattr(pos, "x", None)
        y_val = getattr(pos, "y", None)
        if x_val is None or y_val is None:
            continue
        pt = (float(x_val), float(y_val))
        if not points or points[-1] != pt:
            points.append(pt)
    agent = getattr(dm, "agent", None)
    agent_pos = getattr(agent, "position", None) if agent is not None else None
    start_override = getattr(dm, "start_xy", None)
    if start_override is None and trace:
        first = getattr(trace[0], "position", None)
        if first is not None:
            start_override = (float(getattr(first, "x", 0.0)), float(getattr(first, "y", 0.0)))
    goal_override = getattr(dm, "goal_xy", None)
    if goal_override is None and agent_pos is not None:
        goal_override = (float(getattr(agent_pos, "x", 0.0)), float(getattr(agent_pos, "y", 0.0)))
    if agent_pos is not None:
        terminal = (float(getattr(agent_pos, "x", 0.0)), float(getattr(agent_pos, "y", 0.0)))
        if not points or points[-1] != terminal:
            points.append(terminal)
    if not points:
        return
    rx = [pt[0] for pt in points]
    ry = [pt[1] for pt in points]
    if len(rx) != len(ry):
        return
    try:
        _save_path_overlay(dm, rx, ry, algo_label, start_override, goal_override)
    except Exception:
        pass


__all__ = [
    'WrappedAStar', 'WrappedPRM', 'WrappedPRMv2', 'WrappedRRTv4RRT', 'WrappedRRTv4Connect', 'WrappedRRTv4Dynamic',
    'WrappedDijkstra', 'WrappedThetaStar', 'WrappedPotentialField', 'save_path_overlay_from_trace'
]
