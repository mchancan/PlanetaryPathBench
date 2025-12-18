"""
RRT_CONNECT_2D for Occupancy Map
based on rrt_connect.py and rrt_occupancy.py
"""

import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports from this directory (same pattern as rrt_occupancy)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import env_occupancy
import plotting_occupancy
import utils_occupancy


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class RrtConnectOccupancy:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max, occupancy_map_path):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]

        self.env = env_occupancy.EnvOccupancy(occupancy_map_path)
        self.plotting = plotting_occupancy.PlottingOccupancy(s_start, s_goal, self.env.occupancy_map)
        self.utils = utils_occupancy.UtilsOccupancy(self.env.occupancy_map)

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.occupancy_map = self.env.occupancy_map

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.V1, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new)

                if node_new_prim and not self.utils.is_collision(node_new_prim, node_near_prim):
                    self.V2.append(node_new_prim)

                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new)
                        if node_new_prim2 and not self.utils.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                if self.is_node_same(node_new_prim, node_new):
                    return self.extract_path(node_new, node_new_prim)

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid

        return None

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node((node_new_prim2.x, node_new_prim2.y))
        node_new.parent = node_new_prim
        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        return (node_new_prim.x == node_new.x) and (node_new_prim.y == node_new.y)

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

    @staticmethod
    def extract_path(node_new, node_new_prim):
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new
        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim
        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        return list(list(reversed(path1)) + path2)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    x_start = (50, 30)
    x_goal = (150, 120)
    occupancy_map_path = "/home/mv/mp/PathPlanning/Sampling_based_Planning/rrt_2D/data/intel_inv.npy"

    # show map briefly
    if os.path.exists(occupancy_map_path):
        m = np.load(occupancy_map_path)
        plt.imshow(m); plt.title("Occupancy map"); plt.colorbar(); plt.show()
    else:
        # create simple sample map if missing
        sample_map = np.zeros((200, 200))
        sample_map[60:100, 40:80] = 1
        np.save(occupancy_map_path, sample_map)

    rrt_conn = RrtConnectOccupancy(x_start, x_goal, 0.8, 0.05, 5000, occupancy_map_path)
    path = rrt_conn.planning()

    if path:
        # Plot using the occupancy plotting.animation(...) method.
        # Combine both trees' vertices so the single animation function can draw them.
        all_vertices = rrt_conn.V1 + rrt_conn.V2
        rrt_conn.plotting.animation(all_vertices, path, "RRT_CONNECT (Occupancy)", True)
    else:
        print("No Path Found!")


if __name__ == '__main__':
    main()