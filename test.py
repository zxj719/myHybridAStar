import numpy as np
import math
import sys
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial as kd
import reeds_shepp as rsCurve
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

class Map:
    def __init__(self):
        self.obstacleX, self.obstacleY = [], []
        self.generate_obstacles()

    def generate_obstacles(self):
        # Create boundary obstacles
        self.create_line(0, 0, 50, 0)   # Bottom boundary
        self.create_line(0, 0, 0, 50)   # Left boundary
        self.create_line(0, 50, 50, 50) # Top boundary
        self.create_line(50, 0, 50, 50) # Right boundary
        
        # Create internal obstacles
        self.create_line(6, 30, 19, 30)   # Horizontal line segment
        self.create_line(30, 30, 50, 30)  # Horizontal line segment
        self.create_line(20, 0, 20, 30)   # Vertical line segment
        self.create_line(30, 0, 30, 30)   # Vertical line segment
        self.create_line(15, 40, 15, 49)  # Vertical line segment
        self.create_line(25, 35, 39, 35)  # Horizontal line segment
        
        # Create a complex internal structure
        # self.create_rectangle(10, 10, 15, 15)  # Rectangle
        # self.create_circle(25, 25, 5)  # Circle

    def create_line(self, x1, y1, x2, y2):
        if x1 == x2:  # Vertical line
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.obstacleX.append(x1)
                self.obstacleY.append(y)
        elif y1 == y2:  # Horizontal line
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.obstacleX.append(x)
                self.obstacleY.append(y1)

    def create_rectangle(self, x1, y1, x2, y2):
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                self.obstacleX.append(x)
                self.obstacleY.append(y)

    def create_circle(self, cx, cy, r):
        for x in range(cx - r, cx + r + 1):
            for y in range(cy - r, cy + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    self.obstacleX.append(x)
                    self.obstacleY.append(y)

    def build_kd_tree(self):
        obstacles = np.vstack((self.obstacleX, self.obstacleY)).T
        self.kd_tree = KDTree(obstacles)

    def query_obstacles(self, point, radius):
        return self.kd_tree.query_ball_point(point, radius)

# 示例使用
map = Map()

plt.plot(map.obstacleX, map.obstacleY, "sk")
plt.show()
