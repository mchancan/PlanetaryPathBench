"""Adapter for RRTv4 RRT planners (occupancy-aware files in RRTv4/).

This adapter accepts a numpy occupancy map (or a path) and calls the
`RrtOccupancy` class in `RRTv4/rrt_occupancy.py`. The RRTv4 code expects a
filesystem path to the .npy map, so when given an array we write a temp file.
"""
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

    occ = None
    cleanup_temp = False
    occ_shape = None
    occ_ratio = None
    # If map_array is a path string, use it directly; otherwise write temp .npy
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

    # Defaults scale with grid size but can be overridden.
    step_len = float(kwargs.get('step_len', default_step))
    goal_sample_rate = float(kwargs.get('goal_sample_rate', default_goal_rate))
    iter_max = int(kwargs.get('iter_max', default_iters))

    # Import planner class (try package then relative)
    try:
        from RRTv4.rrt_occupancy import RrtOccupancy
    except Exception:
        from PathPlanning.RRTv4.rrt_occupancy import RrtOccupancy

    planner = RrtOccupancy((sx, sy), (gx, gy), step_len, goal_sample_rate, iter_max, map_path)

    with suppress_plots():
        try:
            path = planner.planning()
        except Exception:
            path = None

    # Clean up temp file if we created one
    try:
        if cleanup_temp and os.path.exists(map_path):
            os.remove(map_path)
    except Exception:
        pass

    if not path:
        return [sx], [sy]

    # RRTv4 returns path as list of (x,y) start->goal already for rrt_occupancy
    rx = [p[0] for p in path]
    ry = [p[1] for p in path]
    return rx, ry
