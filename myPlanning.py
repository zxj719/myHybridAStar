import math
import sys
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd
import reeds_shepp as rsCurve
from dataclasses import dataclass

class Car:
    maxSteerAngle = 0.6
    steerPresion = 10
    wheelBase = 3.5
    axleToFront = 4.5
    axleToBack = 1
    width = 3

class Cost:
    reverse = 10
    directionChange = 150
    steerAngle = 1
    steerAngleChange = 5
    hybridCost = 50

class Map:

    xyResolution: int = 4     # grid block length
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

class Node:
    def __init__(self, gridIndex, traj, steeringAngle, direction, cost, parentIndex):
        self.gridIndex = gridIndex         # grid block x, y, yaw index
        self.traj = traj                   # trajectory x, y  of a simulated node
        self.steeringAngle = steeringAngle # steering angle throughout the trajectory
        self.direction = direction         # direction throughout the trajectory
        self.cost = cost                   # node cost
        self.parentIndex = parentIndex     # parent node index

# 坐标节点
@dataclass(eq=False)
class Node:
    x: float
    y: float
    angle: float
    dir: int
    G: float = 0.        # G代价
    cost: float = None   # F代价 = G + H
    parent: Node = None # 父节点指针

    def __post_init__(self):
        # Grid index
        self.x_idx, self.y_idx, self.yaw_idx= Map.config_idx(self.x, self.y, self.angle)
        if self.cost is None:
            self.cost = self.heuristic([self.x, self.y], [map.g[0], map.g[1]])
    
    def __call__(self, u, dt):
        # 生成新节点 -> new_node = node(u, dt)
        x, y, yaw = motion_model([self.x, self.y, self.angle], u, dt)
        G = self.G + self.heuristic([self.x, self.y], [x, y]) + abs(yaw - self.yaw)
        return Node(x, y, yaw, G, parent=self)
        
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
       
    def heuristic(self, TARG = END):
        """启发搜索, 计算启发值H并更新F值"""
        H = self.heuristic([self.x, self.y], TARG)
        self.cost = self.G + H
        return H

    def is_end(self, err = ERR):
        """是否终点, 启发值H小于err"""
        if self.cost - self.G < err:
            return True
        return False
    
    def in_map(self, map_array = MAP.map_array):
        """是否在地图中"""
        return (0 <= self.x < map_array.shape[1]) and (0 <= self.y < map_array.shape[0]) # h*w维, 右边不能取等!!!

    def is_collided(self, map_array = MAP.map_array):
        """是否发生碰撞"""
        # 计算车辆的边界框的四个顶点坐标
        cos_ = math.cos(self.yaw)
        sin_ = math.sin(self.yaw)
        LC = CAR_LENGTH/2 * cos_
        LS = CAR_LENGTH/2 * sin_
        WC = CAR_WIDTH/2 * cos_
        WS = CAR_WIDTH/2 * sin_
        x1 = self.x + LC + WS
        y1 = self.y - LS + WC
        x2 = self.x + LC - WS
        y2 = self.y - LS - WC
        x3 = self.x - LC + WS
        y3 = self.y + LS + WC
        x4 = self.x - LC - WS
        y4 = self.y + LS - WC
        # 检查边界框所覆盖的栅格是否包含障碍物和出界
        for i in range(int(min([x1, x2, x3, x4])/MAP_NORM), int(max([x1, x2, x3, x4])/MAP_NORM)):
            for j in range(int(min([y1, y2, y3, y4])/MAP_NORM), int(max([y1, y2, y3, y4])/MAP_NORM)):
                if i < 0 or i >= map_array.shape[1]:
                    return True
                if j < 0 or j >= map_array.shape[0]:
                    return True
                if map_array[j, i] == 0: # h*w维, y是第一个索引, 0表示障碍物
                    return True
        return False
        
    @staticmethod
    def heuristic(P1, P2):
        """Euclid Distance"""
        return math.hypot(P1[0] - P2[0], P1[1] - P2[1])

class HolonomicNode:
    def __init__(self, gridIndex, cost, parentIndex):
        self.gridIndex = gridIndex
        self.cost = cost
        self.parentIndex = parentIndex


def index(Node):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([Node.gridIndex[0], Node.gridIndex[1], Node.gridIndex[2]])

def motionCommands():
    # Motion commands for a Non-Holonomic Robot like a Car or Bicycle (Trajectories using Steer Angle and Direction)
    direction = 1
    motionCommand = []
    for i in np.arange(Car.maxSteerAngle, -(Car.maxSteerAngle + Car.maxSteerAngle/Car.steerPresion), -Car.maxSteerAngle/Car.steerPresion):
        motionCommand.append([i, direction])
        motionCommand.append([i, -direction])
    return motionCommand

def holonomicMotionCommands():
    # Action set for a Point/Omni-Directional/Holonomic Robot (8-Directions)
    holonomicMotionCommand = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    return holonomicMotionCommand

def kinematicSimulationNode(currentNode, motionCommand, map, simulationLength=4, step = 0.8 ):
    # Simulate node using given current Node and Motion Commands
    traj = []
    angle = rsCurve.pi_2_pi(currentNode.traj[-1][2] + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))
    traj.append([currentNode.traj[-1][0] + motionCommand[1] * step * math.cos(angle),
                currentNode.traj[-1][1] + motionCommand[1] * step * math.sin(angle),
                rsCurve.pi_2_pi(angle + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))])
    for i in range(int((simulationLength/step))-1):
        traj.append([traj[i][0] + motionCommand[1] * step * math.cos(traj[i][2]),
                    traj[i][1] + motionCommand[1] * step * math.sin(traj[i][2]),
                    rsCurve.pi_2_pi(traj[i][2] + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))])

    # Find grid index
    gridIndex = [round(traj[-1][0]/map.xyResolution), round(traj[-1][1]/map.xyResolution), round(traj[-1][2]/map.yawResolution)]

    # Check if node is valid
    if not isValid(traj, gridIndex, map):
        return None

    # Calculate Cost of the node
    cost = simulatedPathCost(currentNode, motionCommand, simulationLength)

    return Node(gridIndex, traj, motionCommand[0], motionCommand[1], cost, index(currentNode))

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

def isValid(traj, gridIndex, map):
    # Check if Node is out of map bounds
    if gridIndex[0]<=map.mapMinX or gridIndex[0]>=map.mapMaxX or \
       gridIndex[1]<=map.mapMinY or gridIndex[1]>=map.mapMaxY:
        return False

    # Check if Node is colliding with an obstacle
    if collision(traj, map):
        return False
    return True

def collision(traj, map):
    carRadius = (Car.axleToFront + Car.axleToBack)/2 + 1
    dl = (Car.axleToFront - Car.axleToBack)/2
    for i in traj:
        cx = i[0] + dl * math.cos(i[2])
        cy = i[1] + dl * math.sin(i[2])
        pointsInObstacle = map.ObstacleKDTree.query_ball_point([cx, cy], carRadius)
        if not pointsInObstacle:
            continue

        for p in pointsInObstacle:
            xo = map.obstacleX[p] - cx
            yo = map.obstacleY[p] - cy
            dx = xo * math.cos(i[2]) + yo * math.sin(i[2])
            dy = -xo * math.sin(i[2]) + yo * math.cos(i[2])

            if abs(dx) < carRadius and abs(dy) < Car.width / 2 + 1:
                return True

    return False

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

def simulatedPathCost(currentNode, motionCommand, simulationLength):
    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    if motionCommand[1] == 1:
        cost += simulationLength 
    else:
        cost += simulationLength * Cost.reverse

    # Direction change cost
    if currentNode.direction != motionCommand[1]:
        cost += Cost.directionChange

    # Steering Angle Cost
    cost += motionCommand[0] * Cost.steerAngle

    # Steering Angle change cost
    cost += abs(motionCommand[0] - currentNode.steeringAngle) * Cost.steerAngleChange

    return cost

def eucledianCost(holonomicMotionCommand):
    # Compute Eucledian Distance between two nodes
    return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])

def holonomicNodeIndex(HolonomicNode):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([HolonomicNode.gridIndex[0], HolonomicNode.gridIndex[1]])

def obstaclesMap(obstacleX, obstacleY, xyResolution):
    # Compute Grid Index for obstacles
    obstacleX = [round(x / xyResolution) for x in obstacleX]
    obstacleY = [round(y / xyResolution) for y in obstacleY]

    # Set all Grid locations to No Obstacle
    obstacles =[[False for i in range(max(obstacleY))]for i in range(max(obstacleX))]

    # Set Grid Locations with obstacles to True
    for x in range(max(obstacleX)):
        for y in range(max(obstacleY)):
            for i, j in zip(obstacleX, obstacleY):
                if math.hypot(i-x, j-y) <= 1/2:
                    obstacles[i][j] = True
                    break
    return obstacles

def holonomicNodeIsValid(neighbourNode, obstacles, map):
    # Check if Node is out of map bounds
    if neighbourNode.gridIndex[0]<= map.mapMinX or \
       neighbourNode.gridIndex[0]>= map.mapMaxX or \
       neighbourNode.gridIndex[1]<= map.mapMinY or \
       neighbourNode.gridIndex[1]>= map.mapMaxY:
        return False

    # Check if Node on obstacle
    if obstacles[neighbourNode.gridIndex[0]][neighbourNode.gridIndex[1]]:
        return False

    return True

def holonomicCostsWithObstacles(goalNode, map):
    gridIndex = [round(goalNode.traj[-1][0]/map.xyResolution), round(goalNode.traj[-1][1]/map.xyResolution)]
    gNode =HolonomicNode(gridIndex, 0, tuple(gridIndex))

    obstacles = obstaclesMap(map.obstacleX, map.obstacleY, map.xyResolution)

    holonomicMotionCommand = holonomicMotionCommands()

    openSet = {holonomicNodeIndex(gNode): gNode}
    closedSet = {}

    priorityQueue =[]
    heapq.heappush(priorityQueue, (gNode.cost, holonomicNodeIndex(gNode)))

    while True:
        if not openSet:
            break

        _, currentNodeIndex = heapq.heappop(priorityQueue)
        currentNode = openSet[currentNodeIndex]
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode

        for i in range(len(holonomicMotionCommand)):
            neighbourNode = HolonomicNode([currentNode.gridIndex[0] + holonomicMotionCommand[i][0], currentNode.gridIndex[1] + holonomicMotionCommand[i][1]],
                                      currentNode.cost + eucledianCost(holonomicMotionCommand[i]), currentNodeIndex)

            if not holonomicNodeIsValid(neighbourNode, obstacles, map):
                continue

            neighbourNodeIndex = holonomicNodeIndex(neighbourNode)

            if neighbourNodeIndex not in closedSet:            
                if neighbourNodeIndex in openSet:
                    if neighbourNode.cost < openSet[neighbourNodeIndex].cost:
                        openSet[neighbourNodeIndex].cost = neighbourNode.cost
                        openSet[neighbourNodeIndex].parentIndex = neighbourNode.parentIndex
                        # heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
                else:
                    openSet[neighbourNodeIndex] = neighbourNode
                    heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))

    holonomicCost = [[np.inf for i in range(max(map.obstacleY))]for i in range(max(map.obstacleX))]

    for nodes in closedSet.values():
        holonomicCost[nodes.gridIndex[0]][nodes.gridIndex[1]]=nodes.cost

    return holonomicCost



def backtrack(startNode, goalNode, closedSet):

    # Goal Node data
    startNodeIndex= index(startNode)
    currentNodeIndex = goalNode.parentIndex
    currentNode = closedSet[currentNodeIndex]
    x=[]
    y=[]
    yaw=[]

    # Iterate till we reach start node from goal node
    while currentNodeIndex != startNodeIndex:
        a, b, c = zip(*currentNode.traj)
        x += a[::-1] 
        y += b[::-1] 
        yaw += c[::-1]
        currentNodeIndex = currentNode.parentIndex
        currentNode = closedSet[currentNodeIndex]
    return x[::-1], y[::-1], yaw[::-1]

def run(s, g, map, plt):
    # Compute Grid Index for start and Goal node
    sGridIndex = [round(s[0] / map.xyResolution), \
                  round(s[1] / map.xyResolution), \
                  round(s[2]/map.yawResolution)]
    gGridIndex = [round(g[0] / map.xyResolution), \
                  round(g[1] / map.xyResolution), \
                  round(g[2]/map.yawResolution)]

    # Generate all Possible motion commands to car
    motionCommand = motionCommands()

    # Create start and end Node
    startNode = Node(sGridIndex, [s], 0, 1, 0 , tuple(sGridIndex))
    goalNode = Node(gGridIndex, [g], 0, 1, 0, tuple(gGridIndex))

    # Find Holonomic Heuristric
    holonomicHeuristics = holonomicCostsWithObstacles(goalNode, map)

    # Add start node to open Set
    openSet = {index(startNode):startNode}
    closedSet = {}

    # Create a priority queue for acquiring nodes based on their cost's
    costQueue = heapdict()

    # Add start mode into priority queue
    costQueue[index(startNode)] = max(startNode.cost , Cost.hybridCost * holonomicHeuristics[startNode.gridIndex[0]][startNode.gridIndex[1]])
    counter = 0

    # Run loop while path is found or open set is empty
    while True:
        counter +=1
        # Check if openSet is empty, if empty no solution available
        if not openSet:
            return None

        # Get first node in the priority queue
        currentNodeIndex = costQueue.popitem()[0]
        currentNode = openSet[currentNodeIndex]

        # Revove currentNode from openSet and add it to closedSet
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode

        # Get Reed-Shepp Node if available
        rSNode = reedsSheppNode(currentNode, goalNode, map)

        # Id Reeds-Shepp Path is found exit
        if rSNode:
            closedSet[index(rSNode)] = rSNode
            break

        # USED ONLY WHEN WE DONT USE REEDS-SHEPP EXPANSION OR WHEN START = GOAL
        if currentNodeIndex == index(goalNode):
            print("Path Found")
            print(currentNode.traj[-1])
            break

        # Get all simulated Nodes from current node
        for i in range(len(motionCommand)):
            simulatedNode = kinematicSimulationNode(currentNode, motionCommand[i], map)

            # Check if path is within map bounds and is collision free
            if not simulatedNode:
                continue

            # Draw Simulated Node
            x,y,z =zip(*simulatedNode.traj)
            plt.plot(x, y, linewidth=0.3, color='g')

            # Check if simulated node is already in closed set
            simulatedNodeIndex = index(simulatedNode)
            if simulatedNodeIndex not in closedSet: 

                # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                if simulatedNodeIndex not in openSet:
                    openSet[simulatedNodeIndex] = simulatedNode
                    costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
                else:
                    if simulatedNode.cost < openSet[simulatedNodeIndex].cost:
                        openSet[simulatedNodeIndex] = simulatedNode
                        costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
    
    # Backtrack
    x, y, yaw = backtrack(startNode, goalNode, closedSet)

    return x, y, yaw

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
    # obstacleX, obstacleY = map()

    # Calculate map Paramaters
    # map = calculatemap(obstacleX, obstacleY, 4, np.deg2rad(15.0))

    # Run Hybrid A*
    x, y, yaw = run(s, g, map, plt)

    # Draw Start, Goal Location Map and Path
    # plt.arrow(s[0], s[1], 1*math.cos(s[2]), 1*math.sin(s[2]), width=.1)
    # plt.arrow(g[0], g[1], 1*math.cos(g[2]), 1*math.sin(g[2]), width=.1)
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