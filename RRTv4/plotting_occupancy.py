"""
Plotting tools for RRT with Occupancy Map
@author: huiming zhou (modified for occupancy maps)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class PlottingOccupancy:
    def __init__(self, x_start, x_goal, occupancy_map):
        """
        Initialize plotting for occupancy map
        
        Args:
            x_start (tuple): Start position (x, y)
            x_goal (tuple): Goal position (x, y)
            occupancy_map (numpy.ndarray): 2D occupancy grid
        """
        self.xI, self.xG = x_start, x_goal
        self.occupancy_map = occupancy_map
        self.height, self.width = occupancy_map.shape

    def animation(self, nodelist, path, name, animation=False):
        """
        Animate the RRT algorithm
        
        Args:
            nodelist (list): List of nodes in the tree
            path (list): Final path from start to goal
            name (str): Title for the plot
            animation (bool): Whether to animate the tree growth
        """
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)

    def plot_grid(self, name):
        """
        Plot the occupancy grid
        
        Args:
            name (str): Title for the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Display the occupancy map
        # Invert y-axis to match typical coordinate system (origin at bottom-left)
        ax.imshow(self.occupancy_map, cmap='gray_r', origin='lower', 
                  extent=[0, self.width, 0, self.height])
        
        # Plot start and goal positions
        ax.plot(self.xI[0], self.xI[1], "bs", markersize=10, label='Start')
        ax.plot(self.xG[0], self.xG[1], "rs", markersize=10, label='Goal')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Store the axis for later use
        self.ax = ax

    @staticmethod
    def plot_visited(nodelist, animation):
        """
        Plot the visited nodes and tree structure
        
        Args:
            nodelist (list): List of nodes in the tree
            animation (bool): Whether to animate the tree growth
        """
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], 
                            "-g", alpha=0.6, linewidth=1)
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                               lambda event:
                                               [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], 
                            "-g", alpha=0.6, linewidth=1)

    @staticmethod
    def plot_path(path):
        """
        Plot the final path
        
        Args:
            path (list): List of (x, y) coordinates forming the path
        """
        if len(path) != 0:
            path_x = [x[0] for x in path]
            path_y = [x[1] for x in path]
            plt.plot(path_x, path_y, '-r', linewidth=3, label='Path')
            plt.legend()
            plt.pause(0.01)
        plt.show()

    def plot_occupancy_map_only(self, name="Occupancy Map"):
        """
        Plot only the occupancy map without any path
        
        Args:
            name (str): Title for the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display the occupancy map
        ax.imshow(self.occupancy_map, cmap='gray_r', origin='lower',
                  extent=[0, self.width, 0, self.height])
        
        # Plot start and goal positions
        ax.plot(self.xI[0], self.xI[1], "bs", markersize=10, label='Start')
        ax.plot(self.xG[0], self.xG[1], "rs", markersize=10, label='Goal')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.show()
