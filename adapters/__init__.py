"""Adapter registry for planners.

Each adapter module must expose a `run(map_array, start, goal, **kwargs) -> (rx, ry)` function.
Register adapters in the ADAPTERS dict below by their short name.
"""
from . import a_star, rrt
from . import dstar, dijkstra, thetastar
from . import rrtv4_rrt, rrtv4_connect, rrtv4_dynamic

# Only include adapters that have been smoke-tested in this run.
ADAPTERS = {
    'a_star': a_star,
    'astar': a_star,
    'dstar': dstar,
    'rrtv4_rrt': rrtv4_rrt,
    'rrtv4_connect': rrtv4_connect,
    'rrtv4_dynamic': rrtv4_dynamic,
    'dijkstra': dijkstra,
    'thetastar': thetastar,
}

def get_adapter(name):
    return ADAPTERS.get(name)
