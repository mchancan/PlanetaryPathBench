"""Adapter for DStar planner."""
from typing import Tuple
import math

def run(map_array, start, goal, **kwargs):
    try:
        from DStar.dstar import Map, Dstar
    except Exception:
        from PathPlanning.DStar.dstar import Map, Dstar

    resolution = kwargs.get('resolution', 1.0)
    origin = kwargs.get('origin', (0.0, 0.0))
    occupied_value = kwargs.get('occupied_value', 1)

    m = Map(0, 0, occ_map=map_array, resolution=resolution, origin=origin, occupied_value=occupied_value)

    # Convert start/goal world coords to grid indices
    sx, sy = start
    gx, gy = goal
    sx_i = int(round((sx - origin[0]) / resolution))
    sy_i = int(round((sy - origin[1]) / resolution))
    gx_i = int(round((gx - origin[0]) / resolution))
    gy_i = int(round((gy - origin[1]) / resolution))

    start_state = m.map[sx_i][sy_i]
    end_state = m.map[gx_i][gy_i]

    planner = Dstar(m)
    rx, ry = planner.run(start_state, end_state)

    # rx, ry are in grid indices; convert to world coords
    wx = [origin[0] + (float(x) + 0.5) * resolution for x in rx]
    wy = [origin[1] + (float(y) + 0.5) * resolution for y in ry]
    return wx, wy
