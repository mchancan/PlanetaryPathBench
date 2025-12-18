"""Adapter for RRTv4 Dynamic RRT occupancy planner."""
from adapters._common import suppress_plots, tuned_rrt_params


def run(map_array, start, goal, **kwargs):
    import tempfile
    import numpy as _np
    import os

    def _stats_from_path(path: str):
        mmap = _np.load(path, mmap_mode='r')
        try:
            return tuple(int(v) for v in mmap.shape), float(mmap.mean())
        finally:
            del mmap

    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])

    cleanup_temp = False
    occ_shape = None
    occ_ratio = None
    if isinstance(map_array, str) and os.path.exists(map_array):
        map_path = map_array
        occ_shape, occ_ratio = _stats_from_path(map_path)
    else:
        occ = _np.asarray(map_array)
        if occ.ndim != 2:
            raise ValueError('Occupancy map must be 2D')
        occ_shape = occ.shape
        occ_ratio = float(occ.mean())
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        tf.close()
        map_path = tf.name
        _np.save(map_path, occ)
        cleanup_temp = True

    height, width = occ_shape
    default_step, default_goal_rate, default_iters = tuned_rrt_params((height, width), occ_ratio)

    step_len = float(kwargs.get('step_len', default_step))
    goal_sample_rate = float(kwargs.get('goal_sample_rate', default_goal_rate))
    waypoint_sample_rate = float(kwargs.get('waypoint_sample_rate', 0.5))
    iter_max = int(kwargs.get('iter_max', default_iters))

    try:
        from RRTv4.dynamic_rrt_occupancy import DynamicRrtOccupancy
    except Exception:
        from PathPlanning.RRTv4.dynamic_rrt_occupancy import DynamicRrtOccupancy

    planner = DynamicRrtOccupancy((sx, sy), (gx, gy), step_len, goal_sample_rate,
                                  waypoint_sample_rate, iter_max, map_path)
    with suppress_plots():
        try:
            path = planner.planning()
        except Exception:
            path = None

    # Some RRTv4 planners (dynamic/interactive) keep the final path in
    # an instance attribute `planner.path` instead of returning it. Use
    # that if available so headless adapters can still retrieve the result.
    if not path and hasattr(planner, 'path') and planner.path:
        path = planner.path

    try:
        if cleanup_temp and os.path.exists(map_path):
            os.remove(map_path)
    except Exception:
        pass

    if not path:
        return [sx], [sy]
    rx = [p[0] for p in path]
    ry = [p[1] for p in path]
    return rx, ry
