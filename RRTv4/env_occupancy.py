"""
Environment for RRT with Occupancy Map
@author: huiming zhou (modified for occupancy maps)
"""

import numpy as np
import os


class EnvOccupancy:
    def __init__(self, occupancy_map_path):
        """
        Initialize environment with occupancy map from .npy file
        
        Args:
            occupancy_map_path (str): Path to the .npy file containing the occupancy map
        """
        if not os.path.exists(occupancy_map_path):
            raise FileNotFoundError(f"Occupancy map file not found: {occupancy_map_path}")
        
        # Load occupancy map from .npy file
        self.occupancy_map = np.load(occupancy_map_path)
        
        # Get map dimensions
        self.height, self.width = self.occupancy_map.shape
        
        # Set coordinate ranges (assuming map coordinates start from 0)
        self.x_range = (0, self.width)
        self.y_range = (0, self.height)
        
        print(f"Loaded occupancy map: {self.width}x{self.height}")
        print(f"Obstacle cells: {np.sum(self.occupancy_map)}")
        print(f"Free cells: {np.sum(1 - self.occupancy_map)}")
    
    def is_occupied(self, x, y):
        """
        Check if a coordinate is occupied
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if occupied, False if free
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True  # Out of bounds is considered occupied
        
        return self.occupancy_map[int(y), int(x)] == 1
    
    def get_map_bounds(self):
        """
        Get the bounds of the occupancy map
        
        Returns:
            tuple: (x_min, x_max, y_min, y_max)
        """
        return (0, self.width, 0, self.height)
