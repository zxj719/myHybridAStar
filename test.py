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

class Car:
    maxSteerAngle = 0.6
    steerPresion = 10
    wheelBase = 3.5
    axleToFront = 4.5
    axleToBack = 1
    width = 3
    motion_commands = None

    @classmethod
    def set_motion_commands(cls):
        """Define motion commands"""
        # Motion commands for a Non-Holonomic Robot like a Car (Trajectories using Steer Angle and Direction)
        direction = 1
        motion_commands = []
        for i in np.arange(cls.maxSteerAngle, -(cls.maxSteerAngle + cls.maxSteerAngle/cls.steerPresion), -cls.maxSteerAngle/cls.steerPresion):
            motion_commands.append([i, direction])
            motion_commands.append([i, -direction])
        cls.motion_commands = motion_commands

Car.set_motion_commands()

for i,j in Car.motion_commands:
    print(i,j)
