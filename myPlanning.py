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
from common import SetQueue, GridMap, tic, toc, limit_angle

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

class Cost:
    reverse = 10
    directionChange = 150
    steerAngle = 1
    steerAngleChange = 5
    hybridCost = 50

class Map:

    xyResolution: int = 1     # grid block length
    yawResolution: float = np.deg2rad(15.0)  # grid block possible yaws
    s = [10, 10, np.deg2rad(90)]
    g = [25, 10, np.deg2rad(-90)]

    def __init__(self) -> None:
        # Build Map
        self.obstacleX, self.obstacleY = [], []
        
        for i in range(51):
            self.obstacleX.append(i)
            self.obstacleY.append(0)

        for i in range(51):
            self.obstacleX.append(0)
            self.obstacleY.append(i)

        for i in range(51):
            self.obstacleX.append(i)
            self.obstacleY.append(50)

        for i in range(51):
            self.obstacleX.append(50)
            self.obstacleY.append(i)
        
        for i in range(6,20):
            self.obstacleX.append(i)
            self.obstacleY.append(30)

        for i in range(30,51):
            self.obstacleX.append(i)
            self.obstacleY.append(30) 

        for i in range(0,31):
            self.obstacleX.append(20)
            self.obstacleY.append(i) 

        for i in range(0,31):
            self.obstacleX.append(30)
            self.obstacleY.append(i) 

        for i in range(40,50):
            self.obstacleX.append(15)
            self.obstacleY.append(i)

        for i in range(25,40):
            self.obstacleX.append(i)
            self.obstacleY.append(35)

        # calculate min max map grid index based on obstacles in map
        self.mapMinX, self.mapMinY = self.grid_idx(min(self.obstacleX), min(self.obstacleY))
        self.mapMaxX, self.mapMaxY = self.grid_idx(max(self.obstacleX), max(self.obstacleY))

        # create a KDTree to represent obstacles
        self.ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(self.obstacleX, self.obstacleY)])

    @staticmethod
    def grid_idx(*pos):
        """Index (x,y) into map grid (x_idx, y_idx)"""
        return (round(pos[0] / Map.xyResolution), round(pos[1] / Map.xyResolution))
    
    @staticmethod
    def config_idx(*args):
        """Index configuration (x, y, yaw) as map grid node"""
        return (round(args[0] / Map.xyResolution), round(args[1] / Map.xyResolution), round(args[2] / Map.yawResolution))

map = Map()

@dataclass(eq=False)
class Node:
    x: float
    y: float
    yaw: float
    parent: "Node" = None # 父节点指针
    dir: int = 1
    angle: float = 0
    G: float = 0.        # G代价
    cost: float = None   # F代价 = G + H
    
    def __post_init__(self):
        # Grid index
        self.x_idx, self.y_idx, self.yaw_idx= Map.config_idx(self.x, self.y, self.yaw)
        if self.cost is None:
            self.cost = self.heuristic([self.x, self.y], map.g)
    
    def __call__(self, command):
        # Iterate next node
        x, y, yaw = self.kinematic_simulate(command)
        angle, dir = command
        G = self.G + self.heuristic([self.x, self.y], [x, y]) + self.simulatedPathCost(command)
        return Node(x, y, yaw, parent=self, dir=dir, angle=angle, G=G)
        
    def __eq__(self, other: "Node"):
        # 节点eq比较 -> node in list
        return self.x_idx == other.x_idx and self.y_idx == other.y_idx and self.yaw_idx == other.yaw_idx
        #return self.__hash__() == hash(other)
        
    def __le__(self, other: "Node"):
        # 代价<=比较 -> min(open_list)
        return self.cost <= other.cost
    
    def __lt__(self, other: "Node"):
        # 代价<比较 -> min(open_list)
        return self.cost < other.cost
    
    def __hash__(self) -> int:
        # 节点hash比较 -> node in set
        return hash((self.x_idx, self.y_idx, self.yaw_idx))
       
    def update_cost(self, TARG = map.g):
        """启发搜索, 计算启发值H并更新F值"""
        H = self.heuristic([self.x, self.y], TARG)
        self.cost = self.G + H
        return H

    def is_end(self, err = 0.05):
        """是否终点, 启发值H小于err"""
        if self.cost - self.G < err:
            return True
        return False
    
    def in_map(self):
        """是否在地图中"""
        return (0 <= self.x < max(map.obstacleX)) and (0 <= self.y < max(map.obstacleY))

    def is_collided(self):
        cos_ = math.cos(self.yaw)
        sin_ = math.sin(self.yaw)
        car_len = (Car.axleToFront + Car.axleToBack)/2 + 1
        dl = (Car.axleToFront - Car.axleToBack)/2
        cx = self.x + dl * cos_
        cy = self.y + dl * sin_
        pointsInObstacle = map.ObstacleKDTree.query_ball_point([cx, cy], car_len)
        if pointsInObstacle:
            for p in pointsInObstacle:
                xo = map.obstacleX[p] - cx
                yo = map.obstacleY[p] - cy
                dx = xo * cos_ + yo * sin_
                dy = -xo * sin_ + yo * cos_
                if abs(dx) < car_len and abs(dy) < Car.width / 2 + 1:
                    return True
        return False
        
    @staticmethod
    def heuristic(P1, P2):
        """Euclid Distance"""
        return math.hypot(P1[0] - P2[0], P1[1] - P2[1])
    
    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def kinematic_simulate(self, motionCommand, step=0.8):
        # Simulate node using given current Node and Motion Commands
        x = self.x + motionCommand[1] * step * math.cos(self.yaw)
        y = self.y + motionCommand[1] * step * math.sin(self.yaw)
        yaw = self.wrap_angle(self.yaw + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))
        return x, y, yaw
    
    def simulatedPathCost(self, motionCommand):
        cost = 0
        # Distance cost
        if motionCommand[1] == 1:
            cost += 1
        else:
            cost += Cost.reverse
        # Direction change cost
        if self.dir != motionCommand[1]:
            cost += Cost.directionChange
        # Steering Angle Cost
        cost += motionCommand[0] * Cost.steerAngle
        # Steering Angle change cost
        cost += abs(motionCommand[0] - self.angle) * Cost.steerAngleChange
        return cost



def reedsSheppNode(currentNode, goalNode, map):
    # Get x, y, yaw of currentNode and goalNode
    startX, startY, startYaw = currentNode.traj[-1][0], currentNode.traj[-1][1], currentNode.traj[-1][2]
    goalX, goalY, goalYaw = goalNode.traj[-1][0], goalNode.traj[-1][1], goalNode.traj[-1][2]

    # Instantaneous Radius of Curvature
    radius = math.tan(Car.maxSteerAngle)/Car.wheelBase

    #  Find all possible reeds-shepp paths between current and goal node
    reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 1)

    # Check if reedsSheppPaths is empty
    if not reedsSheppPaths:
        return None

    # Find path with lowest cost considering non-holonomic constraints
    costQueue = heapdict()
    for path in reedsSheppPaths:
        costQueue[path] = reedsSheppCost(currentNode, path)

    # Find first path in priority queue that is collision free
    while len(costQueue)!=0:
        path = costQueue.popitem()[0]
        traj=[]
        traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
        if not collision(traj, map):
            cost = reedsSheppCost(currentNode, path)
            return Node(goalNode.gridIndex ,traj, None, None, cost, index(currentNode))
    return None

def reedsSheppCost(currentNode, path):
    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    for i in path.lengths:
        if i >= 0:
            cost += 1
        else:
            cost += abs(i) * Cost.reverse

    # Direction change cost
    for i in range(len(path.lengths)-1):
        if path.lengths[i] * path.lengths[i+1] < 0:
            cost += Cost.directionChange

    # Steering Angle Cost
    for i in path.ctypes:
        # Check types which are not straight line
        if i!="S":
            cost += Car.maxSteerAngle * Cost.steerAngle

    # Steering Angle change cost
    turnAngle=[0.0 for _ in range(len(path.ctypes))]
    for i in range(len(path.ctypes)):
        if path.ctypes[i] == "R":
            turnAngle[i] = - Car.maxSteerAngle
        if path.ctypes[i] == "WB":
            turnAngle[i] = Car.maxSteerAngle

    for i in range(len(path.lengths)-1):
        cost += abs(turnAngle[i+1] - turnAngle[i]) * Cost.steerAngleChange

    return cost


class HybridAStar:
    def __init__(self, move_step=2, step=0.2):

        self.start = Node(*map.s) 
        self.start.update_cost()
       
        # Error Check
        self.end = Node(*map.g)
        if not self.start.in_map() or not self.end.in_map():
            raise ValueError(f"x坐标y坐标超出地图边界")
        if self.start.is_collided():
            raise ValueError(f"起点x坐标或y坐标在障碍物上")
        if self.end.is_collided():
            raise ValueError(f"终点x坐标或y坐标在障碍物上")
       
        self.reset(move_step)
        
    def reset(self, move_step):
        """重置算法"""
        self.__reset_flag = False
        self.move_step = move_step
        self.close_set = set()                    # 存储已经走过的位置及其G值 
        self.open_queue = SetQueue()              # 存储当前位置周围可行的位置及其F值
        self.path_list = []                       # 存储路径(CloseList里的数据无序)

    def _update_open_list(self, curr: Node):
        """open_list添加可行点"""
        for angle, dir in Car.motion_commands:
            # 更新节点
            next_ = curr
            for _ in range(self.move_step):
                next_ = next_([angle, dir]) # x、y、yaw、G_cost、parent都更新了, F_cost未更新
            
            # 新位置是否在地图外边
            if not next_.in_map():
                continue
            # 新位置是否碰到障碍物
            if next_.is_collided():
                continue
            # 新位置是否在 CloseList 中
            if next_ in self.close_set:
                continue

            # 更新F代价
            H = next_.update_cost()

            # open-list添加/更新结点
            self.open_queue.put(next_)
            
            # 当剩余距离小时, 走慢一点
            if H < 20:
                self.move_step = 1
                
            
    def __call__(self):
        """A*路径搜索"""
        assert not self.__reset_flag, "call之前需要reset"
        print("搜索中\n")

        # 初始化 OpenList
        self.open_queue.put(self.start)

        # 正向搜索节点
        tic()
        while not self.open_queue.empty():
            # 弹出 OpenList 代价 F 最小的点
            curr: Node = self.open_queue.get()
            # 更新 OpenList
            self._update_open_list(curr)
            # 更新 CloseList
            self.close_set.add(curr)
            # 结束迭代
            if curr.is_end():
                break
        print("路径搜索完成\n")
        toc()

        # 节点组合成路径
        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()
            
        # 需要重置
        self.__reset_flag = True

        return self.path_list

#_______________________________________________________________________________________________________________________________________________________
# def run(s, g, map, plt):

#     # Add start node to open Set
#     openSet = {index(startNode):startNode}
#     closedSet = {}

#     # Create a priority queue for acquiring nodes based on their cost's
#     costQueue = heapdict()

#     # Add start mode into priority queue
#     costQueue[index(startNode)] = max(startNode.cost , Cost.hybridCost * holonomicHeuristics[startNode.gridIndex[0]][startNode.gridIndex[1]])
#     counter = 0

#     # Run loop while path is found or open set is empty
#     while True:
#         counter +=1
#         # Check if openSet is empty, if empty no solution available
#         if not openSet:
#             return None

#         # Get first node in the priority queue
#         currentNodeIndex = costQueue.popitem()[0]
#         currentNode = openSet[currentNodeIndex]

#         # Revove currentNode from openSet and add it to closedSet
#         openSet.pop(currentNodeIndex)
#         closedSet[currentNodeIndex] = currentNode

#         # Get Reed-Shepp Node if available
#         rSNode = reedsSheppNode(currentNode, goalNode, map)

#         # Id Reeds-Shepp Path is found exit
#         if rSNode:
#             closedSet[index(rSNode)] = rSNode
#             break

#         # USED ONLY WHEN WE DONT USE REEDS-SHEPP EXPANSION OR WHEN START = GOAL
#         if currentNodeIndex == index(goalNode):
#             print("Path Found")
#             print(currentNode.traj[-1])
#             break

#         # Get all simulated Nodes from current node
#         for i in range(len(motionCommand)):
#             simulatedNode = kinematicSimulationNode(currentNode, motionCommand[i], map)

#             # Check if path is within map bounds and is collision free
#             if not simulatedNode:
#                 continue

#             # Draw Simulated Node
#             x,y,z =zip(*simulatedNode.traj)
#             plt.plot(x, y, linewidth=0.3, color='g')

#             # Check if simulated node is already in closed set
#             simulatedNodeIndex = index(simulatedNode)
#             if simulatedNodeIndex not in closedSet: 

#                 # Check if simulated node is already in open set, if not add it open set as well as in priority queue
#                 if simulatedNodeIndex not in openSet:
#                     openSet[simulatedNodeIndex] = simulatedNode
#                     costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
#                 else:
#                     if simulatedNode.cost < openSet[simulatedNodeIndex].cost:
#                         openSet[simulatedNodeIndex] = simulatedNode
#                         costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
    
#     # Backtrack
#     x, y, yaw = backtrack(startNode, goalNode, closedSet)

#     return x, y, yaw

def drawCar(x, y, yaw, color='black'):
    car = np.array([[-Car.axleToBack, -Car.axleToBack, Car.axleToFront, Car.axleToFront, -Car.axleToBack],
                    [Car.width / 2, -Car.width / 2, -Car.width / 2, Car.width / 2, Car.width / 2]])

    rotationZ = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])
    car = np.dot(rotationZ, car)
    car += np.array([[x], [y]])
    plt.plot(car[0, :], car[1, :], color)

def main():
    # Set Start, Goal x, y, theta
    
    # s = [10, 35, np.deg2rad(0)]
    # g = [22, 28, np.deg2rad(0)]

    # Get Obstacle Map
    obstacleX = map.obstacleX
    obstacleY = map.obstacleY

    # Calculate map Paramaters
    # map = calculatemap(obstacleX, obstacleY, 4, np.deg2rad(15.0))

    # Run Hybrid A*
    x = []
    y = []
    yaw = []
    path = HybridAStar()()
    for node in path:
        x.append(node.x)
        y.append(node.y)
        yaw.append(node.yaw)
    # x, y, yaw = run(s, g, map, plt)

    # Draw Start, Goal Location Map and Path
    # plt.arrow(map.s[0], map.s[1], 1*math.cos(map.s[2]), 1*math.sin(map.s[2]), width=.1)
    # plt.arrow(map.g[0], map.g[1], 1*math.cos(map.g[2]), 1*math.sin(map.g[2]), width=.1)
    # plt.xlim(min(obstacleX), max(obstacleX)) 
    # plt.ylim(min(obstacleY), max(obstacleY))
    # plt.plot(obstacleX, obstacleY, "sk")
    # plt.plot(x, y, linewidth=2, color='r', zorder=0)
    # plt.title("Hybrid A*")


    # Draw Path, Map and Car Footprint
    plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
    plt.plot(obstacleX, obstacleY, "sk")
    for k in np.arange(0, len(x), 2):
        plt.xlim(min(obstacleX), max(obstacleX)) 
        plt.ylim(min(obstacleY), max(obstacleY))
        drawCar(x[k], y[k], yaw[k])
        plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
        plt.title("Hybrid A*")

    # Draw Animated Car
    # for k in range(len(x)):
    #     plt.cla()
    #     plt.xlim(min(obstacleX), max(obstacleX)) 
    #     plt.ylim(min(obstacleY), max(obstacleY))
    #     plt.plot(obstacleX, obstacleY, "sk")
    #     plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
    #     drawCar(x[k], y[k], yaw[k])
    #     plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
    #     plt.title("Hybrid A*")
    #     plt.pause(0.001)
    
    plt.show()

if __name__ == '__main__':
    main()