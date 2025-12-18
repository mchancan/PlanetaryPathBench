"""Adapter for Dijkstra planner."""

from adapters._common import suppress_plots


def run(map_array, start, goal, **kwargs):
    try:
        import Dijkstra.dijkstra as dijkstra_module
    except Exception:
        import PathPlanning.Dijkstra.dijkstra as dijkstra_module

    resolution = kwargs.get('resolution', 1.0)
    robot_radius = kwargs.get('robot_radius', 1.0)
    origin = kwargs.get('origin', (0.0, 0.0))
    occupied_value = kwargs.get('occupied_value', 1)
    show_animation = kwargs.get('show_animation', False)

    dijkstra_module.show_animation = show_animation
    planner = dijkstra_module.DijkstraPlanner([], [], resolution, robot_radius,
                                              occ_map=map_array, origin=origin, occupied_value=occupied_value)

    sx, sy = start[0], start[1]
    gx, gy = goal[0], goal[1]

    with suppress_plots():
        rx, ry = planner.planning(sx, sy, gx, gy)
    return rx, ry
