"""
Utils for collision check with occupancy map
@author: huiming zhou (modified for occupancy maps)
"""

import math
import numpy as np
from rrt_occupancy import Node


class UtilsOccupancy:
    def __init__(self, occupancy_map):
        """
        Initialize utilities with occupancy map
        
        Args:
            occupancy_map (numpy.ndarray): 2D occupancy grid (0=free, 1=occupied)
        """
        self.occupancy_map = occupancy_map
        self.height, self.width = occupancy_map.shape
        self.delta = 0.5  # Safety margin

    def is_collision(self, start, end):
        """
        Check if the path from start to end collides with obstacles
        
        Args:
            start (Node): Start node
            end (Node): End node
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Check if start or end nodes are in obstacles
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True
        
        # Check collision along the path using Bresenham's line algorithm
        return self.line_collision_check(start, end)
    
    def is_inside_obs(self, node):
        """
        Check if a node is inside an obstacle
        
        Args:
            node (Node): Node to check
            
        Returns:
            bool: True if inside obstacle, False otherwise
        """
        x, y = int(node.x), int(node.y)
        
        # Check bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        
        # Check if cell is occupied
        return self.occupancy_map[y, x] == 1
    
    def line_collision_check(self, start, end):
        """
        Check for collisions along a line using Bresenham's algorithm
        
        Args:
            start (Node): Start node
            end (Node): End node
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        x0, y0 = int(start.x), int(start.y)
        x1, y1 = int(end.x), int(end.y)
        
        # Get all points along the line
        points = self.get_line_points(x0, y0, x1, y1)
        
        # Check each point for collision
        for x, y in points:
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return True
            if self.occupancy_map[y, x] == 1:
                return True
        
        return False
    
    def get_line_points(self, x0, y0, x1, y1):
        """
        Get all points along a line using Bresenham's algorithm
        
        Args:
            x0, y0 (int): Start coordinates
            x1, y1 (int): End coordinates
            
        Returns:
            list: List of (x, y) tuples along the line
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        error = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
                
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_step
                
            if error2 < dx:
                error += dx
                y += y_step
        
        return points
    
    @staticmethod
    def get_ray(start, end):
        """
        Get ray parameters for line from start to end
        
        Args:
            start (Node): Start node
            end (Node): End node
            
        Returns:
            tuple: (origin, direction)
        """
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        """
        Calculate Euclidean distance between two nodes
        
        Args:
            start (Node): Start node
            end (Node): End node
            
        Returns:
            float: Distance between nodes
        """
        return math.hypot(end.x - start.x, end.y - start.y)
