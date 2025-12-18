"""Adapter for RRT planner."""
from typing import Tuple
import numpy as np

from adapters._common import ensure_start_goal

def run(map_array: np.ndarray, start, goal, resolution: float = 1.0, robot_radius: float = 0.5, origin=(0.0,0.0), occupied_value=1, rand_area=None, max_iter=500, show_animation=False, **kwargs) -> Tuple[list, list]:
    from RRT.rrt import RRT

    sx, sy, gx, gy = ensure_start_goal(start, goal)

    # choose a sampling area if not provided
    if rand_area is None:
        H, W = map_array.shape
        rand_area = (0.0, max(W * resolution, 1.0))

    rrt = RRT(start=[sx, sy], goal=[gx, gy], obstacle_list=[], rand_area=rand_area, path_resolution=resolution, max_iter=max_iter, robot_radius=robot_radius, occ_map=map_array, origin=origin, occupied_value=occupied_value)

    try:
        import RRT.rrt as _mod
        _mod.show_animation = show_animation
    except Exception:
        pass

    path = rrt.planning(animation=show_animation)
    if path is None:
        return [], []

    rx = [p[0] for p in path]
    ry = [p[1] for p in path]
    return rx, ry
