"""Utilities for converting 2D occupancy numpy arrays into planner-friendly inputs.

Provides fast, vectorized conversions used by multiple planners.
"""
from typing import Tuple, List
import numpy as np
import math


def occupancy_to_ox_oy(occ: np.ndarray, resolution: float = 1.0,
                       origin: Tuple[float, float] = (0.0, 0.0),
                       occupied_value=1, center: bool = True) -> Tuple[List[float], List[float]]:
    """Convert occupancy grid to lists of obstacle x and y coordinates.

    occ: 2D numpy array with shape (H, W). Occupied cells compare equal to occupied_value.
    resolution: size of each cell in world units.
    origin: (origin_x, origin_y) world coordinate of array index (0,0).
    center: if True, use cell centers (col + 0.5) * resolution; otherwise use cell corner.

    Returns (ox, oy) lists of floats.
    """
    occ = np.asarray(occ)
    if occ.ndim != 2:
        raise ValueError('occ must be a 2D array')

    H, W = occ.shape
    rows, cols = np.where(occ == occupied_value)

    if center:
        xs = origin[0] + (cols + 0.5) * resolution
        ys = origin[1] + (rows + 0.5) * resolution
    else:
        xs = origin[0] + (cols) * resolution
        ys = origin[1] + (rows) * resolution

    return xs.astype(float).tolist(), ys.astype(float).tolist()


def occupancy_to_obstacle_list(occ: np.ndarray, resolution: float = 1.0,
                               origin: Tuple[float, float] = (0.0, 0.0),
                               occupied_value=1,
                               obstacle_size: float = None,
                               center: bool = True,
                               cluster: bool = False,
                               cluster_min_area: int = 1,
                               cluster_merge_distance: float = 2.0) -> List[Tuple[float, float, float]]:
    """Convert occupancy grid to a list of circular obstacles [(x,y,size), ...].

    obstacle_size: if None, defaults to resolution * sqrt(2) / 2 (approx. inscribed circle radius)
    """
    if obstacle_size is None:
        obstacle_size = resolution * math.sqrt(2) / 2.0

    ox, oy = occupancy_to_ox_oy(occ, resolution, origin, occupied_value, center)
    # For large maps or when clustering is requested, attempt to produce
    # clustered circular obstacles to reduce obstacle counts for sampling
    # planners (improves runtime and memory).
    try:
        if cluster or (occ.size > 50000):
            from .obstacle_processing import cluster_components_for_sampling
            circles = cluster_components_for_sampling(occ, resolution, origin, occupied_value, min_area_pixels=cluster_min_area, merge_distance=cluster_merge_distance)
            # If clustering produced circles, return them directly
            if circles:
                return circles
    except Exception:
        # If clustering utility not available or fails, fall back to per-cell obstacles
        pass

    return [(x, y, obstacle_size) for x, y in zip(ox, oy)]


def occupancy_to_planner_obstacle_map(occ: np.ndarray, resolution: float = 1.0,
                                     origin: Tuple[float, float] = (0.0, 0.0),
                                     occupied_value=1, center: bool = True):
    """Return planner grid parameters and an obstacle_map in the shape expected by grid planners.

    The obstacle_map returned is a list of lists indexed as obstacle_map[x_index][y_index] (matching
    many planners in this repo).

    Returns: (min_x, min_y, max_x, max_y, x_width, y_width, obstacle_map)
    """
    occ = np.asarray(occ)
    if occ.ndim != 2:
        raise ValueError('occ must be a 2D array')

    H, W = occ.shape

    min_x = float(origin[0])
    min_y = float(origin[1])
    max_x = min_x + W * resolution
    max_y = min_y + H * resolution

    x_width = int(round((max_x - min_x) / resolution))
    y_width = int(round((max_y - min_y) / resolution))

    # Create obstacle_map indexed as [x][y]
    obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]

    rows, cols = np.where(occ == occupied_value)
    for r, c in zip(rows, cols):
        ix = int(c)
        iy = int(r)
        if 0 <= ix < x_width and 0 <= iy < y_width:
            obstacle_map[ix][iy] = True

    return min_x, min_y, max_x, max_y, x_width, y_width, obstacle_map
