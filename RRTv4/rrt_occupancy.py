"""
RRT_2D with Occupancy Map
@author: huiming zhou (modified for occupancy maps)
"""

import os
import sys
import math
import numpy as np

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import env_occupancy
import plotting_occupancy
import utils_occupancy
import matplotlib.pyplot as plt

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class RrtOccupancy:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max, occupancy_map_path):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = env_occupancy.EnvOccupancy(occupancy_map_path)
        self.plotting = plotting_occupancy.PlottingOccupancy(s_start, s_goal, self.env.occupancy_map)
        self.utils = utils_occupancy.UtilsOccupancy(self.env.occupancy_map)

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len and not self.utils.is_collision(node_new, self.s_goal):
                    self.new_state(node_new, self.s_goal)
                    return self.extract_path(node_new)

        return None

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    # Example usage - you can modify these parameters
    """

    x_start = (50, 30)
    x_goal = (150, 120)
    occupancy_map_path = "/home/mv/mp/PathPlanning/Sampling_based_Planning/rrt_2D/data/intel_inv.npy"
    map = np.load(occupancy_map_path)
    
    
    x_start = (10, 10) 
    x_goal = (90, 80)   # Goal node (in grid coordinates) # occ_map.npy
    occupancy_map_path = "/home/mv/mp/PathPlanning/Sampling_based_Planning/rrt_2D/data/occ_map.npy"  # Path to your .npy file
    map = np.load(occupancy_map_path)

    x_start = (100, 100)  # Starting node (in grid coordinates)
    x_goal = (3000, 500)   # Goal node (in grid coordinates) # full_occupancy_map.npy
    occupancy_map_path = "/home/mv/mp/PathPlanning/Sampling_based_Planning/rrt_2D/data/full_occupancy_map.npy"
    map = np.load(occupancy_map_path)//255
    map = np.load(occupancy_map_path)
    map = (-1)*(map-1)
    
    #np.save(occupancy_map_path, map)

    
    x_start = (50, 30)
    x_goal = (150, 120)
    occupancy_map_path = "/home/mv/mp/PathPlanning/Sampling_based_Planning/rrt_2D/data/intel.npy"
    map_ = np.load(occupancy_map_path)
    th = 0.5
    map = (map_ > th)
    map = (-1) * (map - 1)
    """
    
    x_start = (10, 10)  # Starting node (in grid coordinates)
    x_goal = (54, 50)   # Goal node (in grid coordinates) # occupancy_map.npy
    occupancy_map_path = "/home/mv/mp/PathPlanning/Sampling_based_Planning/rrt_2D/data/occupancy_map.npy"  # Path to your .npy file
    map = np.load(occupancy_map_path)

    #np.save(occupancy_map_path, map)
    #print(map.max(), map.min())
    plt.imshow(map)
    plt.colorbar()
    plt.show()
    

    
    # Check if occupancy map file exists, if not create a sample one
    if not os.path.exists(occupancy_map_path):
        print(f"Creating sample occupancy map: {occupancy_map_path}")
        # Create a sample occupancy map (100x100 grid)
        sample_map = np.zeros((100, 100))
        # Add some obstacles
        sample_map[20:40, 30:50] = 1  # Rectangle obstacle
        sample_map[60:80, 20:40] = 1  # Another rectangle
        # Add circular obstacle (approximated)
        for i in range(100):
            for j in range(100):
                if (i-70)**2 + (j-70)**2 <= 15**2:
                    sample_map[i, j] = 1
        np.save(occupancy_map_path, sample_map)
        print("Sample occupancy map created!")

    rrt = RrtOccupancy(x_start, x_goal, 2.0, 0.05, 10000, occupancy_map_path)
    path = rrt.planning()

    if path:
        rrt.plotting.animation(rrt.vertex, path, "RRT with Occupancy Map", True)
    else:
        print("No Path Found!")


if __name__ == '__main__':
    main()
