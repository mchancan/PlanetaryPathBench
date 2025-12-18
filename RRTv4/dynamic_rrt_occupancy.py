"""
DYNAMIC_RRT_2D for Occupancy Map
based on dynamic_rrt.py and rrt_occupancy.py
"""

import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        self.flag = "VALID"


class Edge:
    def __init__(self, n_p, n_c):
        self.parent = n_p
        self.child = n_c
        self.flag = "VALID"


class DynamicRrtOccupancy:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, waypoint_sample_rate, iter_max, occupancy_map_path):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.waypoint_sample_rate = waypoint_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.vertex_old = []
        self.vertex_new = []
        self.edges = []

        self.env = env_occupancy.EnvOccupancy(occupancy_map_path)
        self.plotting = plotting_occupancy.PlottingOccupancy(s_start, s_goal, self.env.occupancy_map)
        self.utils = utils_occupancy.UtilsOccupancy(self.env.occupancy_map)
        self.fig, self.ax = plt.subplots()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.occupancy_map = self.env.occupancy_map
        self.occupancy_map_original = copy.deepcopy(self.env.occupancy_map)
        
        # For dynamic obstacle addition (circular region in occupancy map)
        self.obs_add = [0, 0, 0]  # [x, y, radius]

        self.path = []
        self.waypoint = []

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.edges.append(Edge(node_near, node_new))
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len:
                    self.new_state(node_new, self.s_goal)

                    path = self.extract_path(node_new)
                    self.plot_grid("Dynamic_RRT (Occupancy)")
                    self.plot_visited()
                    self.plot_path(path)
                    self.path = path
                    self.waypoint = self.extract_waypoint(node_new)
                    self.fig.canvas.mpl_connect('button_press_event', self.on_press)
                    plt.show()

                    return

        return None

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x is None or y is None or x < 0 or x >= self.x_range[1] or y < 0 or y >= self.y_range[1]:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            radius = 5  # radius for dynamic obstacle
            print("Add circular obstacle at: x =", x, ", y =", y, ", radius =", radius)
            self.obs_add = [x, y, radius]
            self.add_circle_obstacle(x, y, radius)
            self.utils.update_occupancy_map(self.occupancy_map)
            self.InvalidateNodes()

            if self.is_path_invalid():
                print("Path is Replanning ...")
                path, waypoint = self.replanning()

                print("len_vertex: ", len(self.vertex))
                print("len_vertex_old: ", len(self.vertex_old))
                print("len_vertex_new: ", len(self.vertex_new))

                plt.cla()
                self.plot_grid("Dynamic_RRT (Occupancy)")
                self.plot_vertex_old()
                self.plot_path(self.path, color='blue')
                self.plot_vertex_new()
                self.vertex_new = []
                self.plot_path(path)
                self.path = path
                self.waypoint = waypoint
            else:
                print("Trimming Invalid Nodes ...")
                self.TrimRRT()

                plt.cla()
                self.plot_grid("Dynamic_RRT (Occupancy)")
                self.plot_visited(animation=False)
                self.plot_path(self.path)

            self.fig.canvas.draw_idle()

    def add_circle_obstacle(self, cx, cy, radius):
        """Add a circular obstacle to the occupancy map"""
        h, w = self.occupancy_map.shape
        for i in range(max(0, int(cy - radius - 1)), min(h, int(cy + radius + 2))):
            for j in range(max(0, int(cx - radius - 1)), min(w, int(cx + radius + 2))):
                if math.hypot(j - cx, i - cy) <= radius:
                    self.occupancy_map[i, j] = 1

    def InvalidateNodes(self):
        for edge in self.edges:
            if self.is_collision_obs_add(edge.parent, edge.child):
                edge.child.flag = "INVALID"

    def is_path_invalid(self):
        for node in self.waypoint:
            if node.flag == "INVALID":
                return True
        return False

    def is_collision_obs_add(self, start, end):
        delta = self.utils.delta
        obs_add = self.obs_add

        if math.hypot(start.x - obs_add[0], start.y - obs_add[1]) <= obs_add[2] + delta:
            return True

        if math.hypot(end.x - obs_add[0], end.y - obs_add[1]) <= obs_add[2] + delta:
            return True

        # Check intermediate points along the edge
        dist = math.hypot(end.x - start.x, end.y - start.y)
        if dist < 0.01:
            return False
        
        steps = int(dist / 0.5) + 1
        for i in range(steps):
            t = i / steps
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            if math.hypot(x - obs_add[0], y - obs_add[1]) <= obs_add[2]:
                return True

        return False

    def replanning(self):
        self.TrimRRT()

        for i in range(self.iter_max):
            node_rand = self.generate_random_node_replanning(self.goal_sample_rate, self.waypoint_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.vertex_new.append(node_new)
                self.edges.append(Edge(node_near, node_new))
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len:
                    self.new_state(node_new, self.s_goal)
                    path = self.extract_path(node_new)
                    waypoint = self.extract_waypoint(node_new)
                    print("path: ", len(path))
                    print("waypoint: ", len(waypoint))

                    return path, waypoint

        return None, None

    def TrimRRT(self):
        for i in range(1, len(self.vertex)):
            node = self.vertex[i]
            node_p = node.parent
            if node_p.flag == "INVALID":
                node.flag = "INVALID"

        self.vertex = [node for node in self.vertex if node.flag == "VALID"]
        self.vertex_old = copy.deepcopy(self.vertex)
        self.edges = [Edge(node.parent, node) for node in self.vertex[1:len(self.vertex)]]

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    def generate_random_node_replanning(self, goal_sample_rate, waypoint_sample_rate):
        delta = self.utils.delta
        p = np.random.random()

        if p < goal_sample_rate:
            return self.s_goal
        elif goal_sample_rate < p < goal_sample_rate + waypoint_sample_rate:
            return self.waypoint[np.random.randint(0, len(self.waypoint))]
        else:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

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

    def extract_waypoint(self, node_end):
        waypoint = [self.s_goal]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            waypoint.append(node_now)

        return waypoint

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def plot_grid(self, name):
        # Display the occupancy map
        self.ax.imshow(self.occupancy_map, cmap='Greys', origin='lower', alpha=0.5)
        
        plt.plot(self.s_start.x, self.s_start.y, "bs", linewidth=3)
        plt.plot(self.s_goal.x, self.s_goal.y, "gs", linewidth=3)

        plt.title(name)
        plt.axis("equal")
        plt.xlim(self.x_range)
        plt.ylim(self.y_range)

    def plot_visited(self, animation=True):
        if animation:
            count = 0
            for node in self.vertex:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in self.vertex:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    def plot_vertex_old(self):
        for node in self.vertex_old:
            if node.parent:
                plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    def plot_vertex_new(self):
        count = 0

        for node in self.vertex_new:
            count += 1
            if node.parent:
                plt.plot([node.parent.x, node.x], [node.parent.y, node.y], color='darkorange')
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event:
                                             [exit(0) if event.key == 'escape' else None])
                if count % 10 == 0:
                    plt.pause(0.001)

    @staticmethod
    def plot_path(path, color='red'):
        if path:
            plt.plot([x[0] for x in path], [x[1] for x in path], linewidth=2, color=color)
            plt.pause(0.01)


def main():
    x_start = (50, 30)
    x_goal = (150, 120)
    occupancy_map_path = "/home/mv/mp/PathPlanning/Sampling_based_Planning/rrt_2D/data/intel_inv.npy"

    # Show map briefly
    if os.path.exists(occupancy_map_path):
        m = np.load(occupancy_map_path)
        plt.imshow(m)
        plt.title("Occupancy map")
        plt.colorbar()
        plt.show()
    else:
        # Create simple sample map if missing
        sample_map = np.zeros((200, 200))
        sample_map[60:100, 40:80] = 1
        np.save(occupancy_map_path, sample_map)

    drrt = DynamicRrtOccupancy(x_start, x_goal, 2.0, 0.1, 0.6, 5000, occupancy_map_path)
    drrt.planning()


if __name__ == '__main__':
    main()