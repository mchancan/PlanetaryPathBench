"""Common helpers for adapters.
"""
from collections import deque
from contextlib import contextmanager
from typing import Tuple
import math
import numpy as np


def load_npy_map(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError('Occupancy map must be 2D')
    return arr


def ensure_start_goal(start, goal):
    if len(start) != 2 or len(goal) != 2:
        raise ValueError('start and goal must be (x,y)')
    return float(start[0]), float(start[1]), float(goal[0]), float(goal[1])


@contextmanager
def suppress_plots():
    """Disable ``plt.show``/``plt.pause`` to keep adapters headless."""
    try:
        import matplotlib.pyplot as _plt
    except Exception:
        # Matplotlib not available; nothing to suppress
        yield
        return

    was_interactive = _plt.isinteractive()
    _plt.ioff()

    def _noop(*_args, **_kwargs):
        return None

    orig_show = getattr(_plt, 'show', None)
    orig_pause = getattr(_plt, 'pause', None)
    if orig_show is not None:
        _plt.show = _noop
    if orig_pause is not None:
        _plt.pause = _noop

    try:
        yield
    finally:
        if orig_show is not None:
            _plt.show = orig_show
        if orig_pause is not None:
            _plt.pause = orig_pause
        if was_interactive:
            _plt.ion()


_NEIGHBORS = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)


def _largest_component(occ: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    height, width = occ.shape
    visited = np.zeros_like(occ, dtype=bool)
    largest: Tuple[Tuple[int, int], ...] = ()
    for y in range(height):
        row = occ[y]
        for x in range(width):
            if row[x] != 0 or visited[y, x]:
                continue
            comp = []
            queue = deque([(x, y)])
            visited[y, x] = True
            while queue:
                xc, yc = queue.popleft()
                comp.append((xc, yc))
                for dx, dy in _NEIGHBORS:
                    nx, ny = xc + dx, yc + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx] and occ[ny, nx] == 0:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            if len(comp) > len(largest):
                largest = tuple(comp)
    return largest


def auto_select_start_goal(occ: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Choose deterministic start/goal within the largest free component."""
    if occ.ndim != 2:
        raise ValueError("Occupancy map must be 2D")
    if not np.any(occ == 0):
        raise ValueError("Occupancy map has no free cells")
    component = _largest_component(occ)
    if not component:
        raise ValueError("Unable to locate a free-space component")
    # Deterministic start: lexicographically smallest (y, then x)
    start = min(component, key=lambda p: (p[1], p[0]))
    # Farthest goal in Euclidean distance
    x0, y0 = start
    goal = max(
        component,
        key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2,
    )
    return (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))


def tuned_rrt_params(shape: Tuple[int, int], obstacle_ratio: float = None) -> Tuple[float, float, int]:
    """Return heuristic defaults for RRT-based planners on occupancy grids."""
    height, width = shape
    diag = math.hypot(height, width)

    step_len = max(1.5, min(6.0, diag / 150.0))
    iter_max = max(20000, int(diag * 400))

    if obstacle_ratio is None:
        goal_sample_rate = 0.3
    else:
        clutter = min(0.95, max(0.05, obstacle_ratio))
        goal_sample_rate = min(0.45, max(0.18, 0.5 - 0.4 * clutter))

    return step_len, goal_sample_rate, iter_max
