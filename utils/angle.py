"""Small angle utilities expected by some planners (compat shim).

Provides angle_mod and rot_mat_2d to satisfy imports of `from utils.angle import ...`.
"""
import math
from typing import Union
import numpy as np


def angle_mod(x: Union[float, np.ndarray], zero_2_2pi: bool = False):
    """Normalize angle(s).

    If zero_2_2pi is True, normalize to [0, 2*pi). Otherwise to (-pi, pi].
    Works with scalars and numpy arrays.
    """
    # Compute using numpy to handle arrays or scalars uniformly, then
    # return a Python float for single-value results to avoid surprising
    # numpy-scalar types where callers expect native floats.
    if zero_2_2pi:
        a = np.mod(x, 2 * math.pi)
    else:
        a = np.mod(np.asarray(x) + math.pi, 2 * math.pi) - math.pi

    a_arr = np.asarray(a)
    if a_arr.size == 1:
        return float(a_arr.item())
    return a_arr


def rot_mat_2d(theta: float):
    """Return 2x2 rotation matrix for angle theta."""
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
