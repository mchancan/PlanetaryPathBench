# RRT with Occupancy Maps

This directory contains a modified version of the RRT (Rapidly-exploring Random Tree) algorithm that works with occupancy maps loaded from `.npy` files.

## Files

- `rrt_occupancy.py` - Main RRT algorithm for occupancy maps
- `env_occupancy.py` - Environment class for loading occupancy maps
- `utils_occupancy.py` - Collision detection utilities for occupancy grids
- `plotting_occupancy.py` - Visualization tools for occupancy maps
- `example_usage.py` - Example script showing how to use the algorithm

## Occupancy Map Format

The occupancy map should be a 2D numpy array where:
- `0` represents free space
- `1` represents occupied space (obstacles)
- The map is saved as a `.npy` file

## Usage

### Basic Usage

```python
from rrt_occupancy import RrtOccupancy

# Define start and goal positions
x_start = (10, 10)  # Start position (x, y)
x_goal = (90, 80)   # Goal position (x, y)

# RRT parameters
step_len = 2.0          # Step length for tree expansion
goal_sample_rate = 0.05 # Probability of sampling goal (5%)
iter_max = 10000        # Maximum iterations
occupancy_map_path = "your_map.npy"  # Path to your occupancy map

# Create and run RRT
rrt = RrtOccupancy(x_start, x_goal, step_len, goal_sample_rate, iter_max, occupancy_map_path)
path = rrt.planning()

if path:
    rrt.plotting.animation(rrt.vertex, path, "RRT with Occupancy Map", True)
else:
    print("No path found!")
```

### Running the Example

```bash
# Run the basic example (creates a sample map)
python3 rrt_occupancy.py

# Run the custom example
python3 example_usage.py
```

## Creating Your Own Occupancy Map

```python
import numpy as np

# Create a 100x100 occupancy map
occupancy_map = np.zeros((100, 100))

# Add obstacles
occupancy_map[20:40, 30:50] = 1  # Rectangle obstacle
occupancy_map[60:80, 20:40] = 1  # Another rectangle

# Add circular obstacle (approximated)
for i in range(100):
    for j in range(100):
        if (i-70)**2 + (j-70)**2 <= 15**2:
            occupancy_map[i, j] = 1

# Save the map
np.save("my_map.npy", occupancy_map)
```

## Parameters

- `x_start`: Starting position as (x, y) tuple
- `x_goal`: Goal position as (x, y) tuple  
- `step_len`: Maximum step length for tree expansion
- `goal_sample_rate`: Probability (0-1) of sampling the goal instead of random point
- `iter_max`: Maximum number of iterations
- `occupancy_map_path`: Path to the `.npy` file containing the occupancy map

## Visualization

The algorithm provides:
- Gray background showing the occupancy map (white=free, black=obstacles)
- Blue square for start position
- Red square for goal position
- Green lines showing the tree structure
- Red line showing the final path

## Notes

- Make sure start and goal positions are in free space (not on obstacles)
- The algorithm uses Bresenham's line algorithm for collision detection
- Coordinates are in grid units (integer values)
- The map origin (0,0) is at the bottom-left corner
