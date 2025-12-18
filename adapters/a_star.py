"""Adapter for A* planner that accepts occupancy numpy arrays."""
from typing import Tuple
import numpy as np

from adapters._common import ensure_start_goal, suppress_plots


def run(map_array: np.ndarray, start, goal, resolution: float = 1.0, robot_radius: float = 1.0,
        origin=(0.0, 0.0), occupied_value=1, show_animation=False, **_kwargs) -> Tuple[list, list]:
    try:
        import AStar.a_star as a_star_module
    except Exception:
        import PathPlanning.AStar.a_star as a_star_module

    sx, sy, gx, gy = ensure_start_goal(start, goal)
    a_star_module.show_animation = show_animation
    planner = a_star_module.AStarPlanner([], [], resolution, robot_radius,
                                        occ_map=map_array, origin=origin, occupied_value=occupied_value)

    with suppress_plots():
        rx, ry = planner.planning(sx, sy, gx, gy)
    return rx, ry
