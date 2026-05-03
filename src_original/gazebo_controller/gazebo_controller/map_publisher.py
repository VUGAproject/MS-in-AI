#!/usr/bin/env python3
"""
Static Map Publisher Node
Publishes an OccupancyGrid representation of the maze world for navigation and debugging.
Loads map definition from YAML file specific to each maze.
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np
import yaml
import os
from ament_index_python.packages import get_package_share_directory


class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        
        # Declare parameter for maze folder
        self.declare_parameter('maze', 'basic_maze')
        maze_folder = self.get_parameter('maze').value
        
        # Load map configuration from YAML file
        pkg_share = get_package_share_directory('gazebo_controller')
        map_file = os.path.join(pkg_share, 'sdf', maze_folder, 'map.yaml')
        
        self.get_logger().info(f'Loading map from: {map_file}')
        
        # Check if map file exists
        if not os.path.exists(map_file):
            self.get_logger().warn(f'Map file not found: {map_file}')
            self.get_logger().warn('Map publisher will not publish maps. Create map.yaml to enable map publishing.')
            self.map_available = False
            return
        
        try:
            with open(map_file, 'r') as f:
                map_config = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to load map file: {e}')
            self.get_logger().warn('Map publisher will not publish maps.')
            self.map_available = False
            return
        
        self.map_available = True
        
        # Map parameters from YAML
        self.resolution = map_config['resolution']
        self.width = map_config['width']
        self.height = map_config['height']
        self.origin_x = map_config['origin_x']
        self.origin_y = map_config['origin_y']
        self.walls = map_config['walls']
        
        # Calculate grid dimensions
        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)
        
        self.get_logger().info(f'Creating map: {self.grid_width}x{self.grid_height} cells')
        self.get_logger().info(f'Map dimensions: {self.width}m x {self.height}m')
        self.get_logger().info(f'Found {len(self.walls)} walls')
        
        # Create the occupancy grid
        self.occupancy_grid = self.create_maze_map()
        
        # Publisher
        self.publisher_ = self.create_publisher(OccupancyGrid, '/map', 10)
        
        # Publish at 1 Hz (static map)
        self.timer = self.create_timer(1.0, self.publish_map)
        
        self.get_logger().info('Map publisher initialized')
    
    def create_maze_map(self):
        """Create the occupancy grid based on wall definitions from YAML"""
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        
        for wall in self.walls:
            cx, cy, sx, sy = wall[0], wall[1], wall[2], wall[3]
            rotation = float(wall[4]) if len(wall) > 4 else 0.0
            # Axis-aligned: rotation ≈ 0, ±π
            if abs(rotation) < 0.01 or abs(abs(rotation) - math.pi) < 0.01:
                self.fill_rectangle(grid, cx, cy, sx, sy, 100)
            # 90° rotated: swap dimensions
            elif abs(abs(rotation) - math.pi / 2) < 0.01:
                self.fill_rectangle(grid, cx, cy, sy, sx, 100)
            else:
                self.fill_rotated_rectangle(grid, cx, cy, sx, sy, rotation, 100)
        
        return grid
    
    def fill_rotated_rectangle(self, grid, center_x, center_y, size_x, size_y, rotation, value):
        """Fill a rotated rectangle into the grid by rasterising in the wall's local frame."""
        h, w = grid.shape
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)
        half_x = size_x / 2.0
        half_y = size_y / 2.0
        # Compute axis-aligned bounding box of the rotated rectangle
        bb_hx = half_x * abs(cos_t) + half_y * abs(sin_t)
        bb_hy = half_x * abs(sin_t) + half_y * abs(cos_t)
        min_col = max(0, int((center_x - bb_hx - self.origin_x) / self.resolution))
        max_col = min(w - 1, int((center_x + bb_hx - self.origin_x) / self.resolution) + 1)
        min_row = max(0, int((center_y - bb_hy - self.origin_y) / self.resolution))
        max_row = min(h - 1, int((center_y + bb_hy - self.origin_y) / self.resolution) + 1)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                wx = self.origin_x + (col + 0.5) * self.resolution
                wy = self.origin_y + (row + 0.5) * self.resolution
                dx = wx - center_x
                dy = wy - center_y
                # Rotate to wall-local frame
                lx = dx * cos_t + dy * sin_t
                ly = -dx * sin_t + dy * cos_t
                if abs(lx) <= half_x and abs(ly) <= half_y:
                    grid[row, col] = value

    def fill_rectangle(self, grid, center_x, center_y, size_x, size_y, value):
        """Fill an axis-aligned rectangle in the grid with the given value."""
        # Calculate bounds in world coordinates
        min_x = center_x - size_x / 2.0
        max_x = center_x + size_x / 2.0
        min_y = center_y - size_y / 2.0
        max_y = center_y + size_y / 2.0
        
        # Convert to grid coordinates
        min_col = int((min_x - self.origin_x) / self.resolution)
        max_col = int((max_x - self.origin_x) / self.resolution)
        min_row = int((min_y - self.origin_y) / self.resolution)
        max_row = int((max_y - self.origin_y) / self.resolution)
        
        # Clamp to grid bounds
        min_col = max(0, min_col)
        max_col = min(self.grid_width - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(self.grid_height - 1, max_row)
        
        # Fill the rectangle
        grid[min_row:max_row+1, min_col:max_col+1] = value
    
    def publish_map(self):
        """Publish the occupancy grid map"""
        # Don't publish if map is not available
        if not self.map_available:
            return
            
        msg = OccupancyGrid()
        
        # Header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        # Map metadata
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Flatten the grid (row-major order) and convert to list
        msg.data = self.occupancy_grid.flatten().tolist()
        
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
