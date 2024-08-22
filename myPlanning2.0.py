import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial as kd
import reeds_shepp as rsCurve
from dataclasses import dataclass, field
import time
from matplotlib.ticker import FuncFormatter

IMAGE_PATH = 'map.jpg' 
THRESH = 127              
MAP_SIZE = 10
IMG_SIZE = 400
START =[6.8, 1, np.pi/2]
GOAL = [3, 6.5, np.pi/2]

class Car:
    max_steer_angle = 0.52
    steer_precision = 5
    wheel_base = 0.6
    axle2front = 0.7
    axle2back = 0.2
    width = 0.31
    reverse_cost = 30
    dir_change_cost = 30
    yaw_change_cost = 5
    half_len = (axle2front + axle2back)/2
    half_wid = width / 2
    dl = (axle2front - axle2back)/2
    motion_commands =[[i, dir] for i in np.arange(max_steer_angle, -(max_steer_angle + \
                    max_steer_angle/steer_precision), -max_steer_angle/steer_precision) for dir in [1, -1]]
    
STEP_SIZE = Car.axle2back

class Map:
    def __init__(self, img_path, thresh, size, img_size, yaw_resolution=np.deg2rad(5.0)): 
        
        self.yaw_resolution = yaw_resolution  # grid block possible yaws
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        thresh, map_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY) 
        map_img = cv2.resize(map_img, (img_size, img_size))
        map_img = map_img[::-1]                              
        self.map_array = np.array(map_img)
        self.img_size = self.map_array.shape[0]
        self.size = size
        self.xy_resolution = self.size / self.img_size  # grid block length
        self.obstacle_x, self.obstacle_y = [], []
        self.generate_obstacles()
        
        # create a KDTree to represent obstacles
        self.ObstacleKDTree = kd.KDTree([[x*self.xy_resolution, y*self.xy_resolution] for x, y in zip(self.obstacle_x, self.obstacle_y)])

    def generate_obstacles(self, downsample_factor=2):
        # Use the Sobel operator to detect edges in the binary obstacle map
        sobelx = cv2.Sobel(self.map_array, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.map_array, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize the gradient to binary (edges = 1, no edge = 0)
        gradient_magnitude = (gradient_magnitude > 0).astype(np.uint8)
        
        # Loop through the map array and add only surface obstacle pixels to the KDTree
        for i in range(0, self.img_size, downsample_factor):
            for j in range(0, self.img_size, downsample_factor):
                if gradient_magnitude[j][i] > 0:  # Only consider surface pixels
                    self.obstacle_x.append(i)
                    self.obstacle_y.append(j)

    def config_idx(self, *args):
        """Index configuration (x, y, yaw) as map grid node"""
        return (round(args[0] / self.xy_resolution), round(args[1] / self.xy_resolution), round(args[2] / self.yaw_resolution))

map = Map(IMAGE_PATH, THRESH, MAP_SIZE, IMG_SIZE)

@dataclass(eq=False)
class HybridNode:
    x: float
    y: float
    yaw: float
    parent: "HybridNode" = None 
    dir: int = 1
    angle: float = 0
    G: float = 0.        
    F: float = None
    
    def __post_init__(self):
        # Grid index, update F cost
        self.x_idx, self.y_idx, self.yaw_idx= map.config_idx(self.x, self.y, self.yaw)
        self.F = self.G + self.heuristic(GOAL)
    
    def next_node(self, command, step):
        # Iterate next node
        next_x, next_y, next_yaw = self.kinematic_simulate(command, step)
        angle, dir = command
        G = self.G + self.heuristic([next_x, next_y]) + self.hybrid_cost(command)
        return HybridNode(next_x, next_y, next_yaw, parent=self, dir=dir, angle=angle, G=G)
        
    def __eq__(self, other: "HybridNode"):
        return self.x_idx == other.x_idx and self.y_idx == other.y_idx #and self.yaw_idx == other.yaw_idx
    
    def __lt__(self, other: "HybridNode"):
        return self.F < other.F
    
    def __hash__(self) -> int:
        return hash((self.x_idx, self.y_idx, self.yaw_idx))

    def is_collided(self):
        # out of map?
        if 0 > self.x or self.x >= map.size or 0 > self.y or self.y >= map.size:
            return True
        
        # collides with obstacles?
        cos_ = math.cos(self.yaw)
        sin_ = math.sin(self.yaw)

        cx = self.x + Car.dl * cos_
        cy = self.y + Car.dl * sin_
        pointsInObstacle = map.ObstacleKDTree.query_ball_point([cx, cy], Car.half_len+Car.half_wid)
        if pointsInObstacle:
            for p in pointsInObstacle:
                xo = map.obstacle_x[p]*map.xy_resolution - cx
                yo = map.obstacle_y[p]*map.xy_resolution - cy
                dx = xo * cos_ + yo * sin_
                dy = -xo * sin_ + yo * cos_
                if abs(dx) < Car.half_len and abs(dy) < Car.half_wid+0.05:
                    return True
        return False
 
    
    def heuristic(self, tar):
        """Euclid Distance"""
        return math.hypot(self.x - tar[0], self.y - tar[1])
    
    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def kinematic_simulate(self, command, step):
        # Simulate node using given current HybridNode and Motion Commands
        x = self.x + command[1] * step * math.cos(self.yaw)
        y = self.y + command[1] * step * math.sin(self.yaw)
        yaw = self.wrap_angle(self.yaw + command[1] * step / Car.wheel_base * math.tan(command[0]))
        return x, y, yaw
    
    def hybrid_cost(self, command):
        cost = 0
        # Distance cost
        if command[1] == 1:
            cost += 1
        else:
            cost += Car.reverse_cost
        # Direction change cost
        if self.dir != command[1]:
            cost += Car.dir_change_cost
        # Steering Angle change cost
        cost += abs(command[0] - self.angle) * Car.yaw_change_cost
        return cost
    
def reedsSheppNode(current_node, goal_node):
   # Instantaneous Radius of Curvature
    radius = math.tan(Car.max_steer_angle)/Car.wheel_base*1.2
    # Find all possible reeds-shepp paths between current and goal node
    reedsSheppPaths = rsCurve.calc_all_paths(current_node.x, current_node.y, current_node.yaw, \
                                             goal_node.x, goal_node.y, goal_node.yaw, radius, step_size=STEP_SIZE)
    # Check if reedsSheppPaths is empty
    if not reedsSheppPaths:
        return None
    # Find path with lowest cost considering non-holonomic constraints
    costQueue = heapdict()
    for path in reedsSheppPaths:
        costQueue[path] = reedsSheppCost(current_node, path)
    # Find first path in priority queue that is collision free
    while len(costQueue)!=0:
        path = costQueue.popitem()[0]
        node_list = []
        parent = current_node
        for k in range(len(path.x)):
            node = HybridNode(path.x[k],path.y[k],path.yaw[k], parent)
            if node.is_collided():
                return None
            node_list.append(node)
            parent = node
        node_list.append(HybridNode(goal_node.x, goal_node.y, goal_node.yaw, parent))
        return node_list
    return None

def reedsSheppCost(current_node, path):
    # Previous Node Cost
    cost = current_node.F
    # Distance cost
    for i in path.lengths:
        if i >= 0:
            cost += 1
        else:
            cost += abs(i) * Car.reverse_cost
    # Direction change cost
    for i in range(len(path.lengths)-1):
        if path.lengths[i] * path.lengths[i+1] < 0:
            cost += Car.dir_change_cost
    # Steering Angle change cost
    turnAngle=[0.0 for _ in range(len(path.ctypes))]
    for i in range(len(path.ctypes)):
        if path.ctypes[i] == "R":
            turnAngle[i] = - Car.max_steer_angle
        if path.ctypes[i] == "WB":
            turnAngle[i] = Car.max_steer_angle
    for i in range(len(path.lengths)-1):
        cost += abs(turnAngle[i+1] - turnAngle[i]) * Car.yaw_change_cost
    return cost

class HybridAStar:
    def __init__(self):
        self.start = HybridNode(*START)
        self.close_set = set()                    
        self.open_queue = heapdict()            
        self.path_list = []                       
        self.rSNode = None
        self.closed_list_nodes = []  # 存储所有 closed list 中的节点
        # Error Check
        self.end = HybridNode(*GOAL)
        if self.start.is_collided():
            raise ValueError(f"START on the obstacles!")
        if self.end.is_collided():
            raise ValueError(f"TARGET on the obstacles!")

    def _update_open_list(self, curr: HybridNode):
        # Calculate distance to nearest obstacle
        dist_to_nearest_obstacle, _ = map.ObstacleKDTree.query([curr.x, curr.y])
        
        # Adjust move_step based on distance to obstacles
        step = STEP_SIZE
        if dist_to_nearest_obstacle < STEP_SIZE:
            move_step = 1  # Decrease move_step if close to obstacles
            step = STEP_SIZE/2
        elif dist_to_nearest_obstacle > STEP_SIZE*4:
            move_step = 2  # Increase move_step if far from obstacles
            step = STEP_SIZE*2
        else:
            move_step = 2  # Default move_step
        
        for command in Car.motion_commands:
            next_node = curr
            for _ in range(move_step):
                next_node = next_node.next_node(command, step)
            if next_node.is_collided():
                continue
            if next_node in self.close_set:
                continue
            
            self.open_queue[next_node] = next_node.F
    
    @staticmethod
    def tic(): 
        if 'global_tic_time' not in globals():
            global global_tic_time
            global_tic_time = []
        global_tic_time.append(time.time())

    @staticmethod
    def toc(name='', *, CN=True, digit=6): 
        if 'global_tic_time' not in globals() or not global_tic_time:
            print('no tic' if CN else 'tic not set')  
            return
        name = name+' ' if (name and not CN) else name
        if CN:
            print('%sprocessing %f s。\n' % (name, round(time.time() - global_tic_time.pop(), digit)))
        else:
            print('%sElapsed time is %f seconds.\n' % (name, round(time.time() - global_tic_time.pop(), digit)))
                
    def run(self):
        """A* path search"""
        print("Searching path...\n")
        # Initialize OpenList
        self.open_queue[self.start] = self.start.F
        # Start forward search
        self.tic()
        while self.open_queue:
            # Pop the node with the minimum F cost from OpenList
            curr: HybridNode = self.open_queue.popitem()[0]
            self.closed_list_nodes.append(curr)  # 保存当前节点到 closed list

            # Get Reed-Shepp node if available
            rSNode = reedsSheppNode(curr, self.end)
            # If Reed-Shepp path is found, exit
            if rSNode:
                self.close_set = set()
                for i in range(len(rSNode)):
                    self.close_set.add(rSNode[i])
                curr = rSNode[-1]
                break
            # Update OpenList
            self._update_open_list(curr)
            # Update CloseList
            self.close_set.add(curr)
        print("Done.\n")
        self.toc()

        # Combine nodes to form the path
        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()
        return self.path_list, self.closed_list_nodes  

def draw_car(ax, x, y, yaw, color='black'):
    car = np.array([[-Car.axle2back, -Car.axle2back, Car.axle2front, Car.axle2front, -Car.axle2back],
                    [Car.width / 2, -Car.width / 2, -Car.width / 2, Car.width / 2, Car.width / 2]])

    rotation_z = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])
    car = np.dot(rotation_z, car)
    car = np.array([[x], [y]]) + np.round(car / map.xy_resolution)
    ax.plot(car[0, :], car[1, :], color)

def scale(x, pos):
        return '{:.1f}'.format(x*map.xy_resolution)

def plot(x, y, yaw, closed_list_nodes):
    # Figure 1: Draw Animated Car
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for show_time in range(1):
        for k in range(len(x)):
            ax1.cla()
            ax1.set_xlim(0, map.img_size) 
            ax1.xaxis.set_major_formatter(FuncFormatter(scale))
            ax1.set_ylim(0, map.img_size)
            ax1.yaxis.set_major_formatter(FuncFormatter(scale))
            ax1.plot(map.obstacle_x, map.obstacle_y, "sk")
            ax1.plot(x, y, linewidth=0.02/map.xy_resolution, color='r', zorder=0)
            draw_car(ax1, x[k], y[k], yaw[k])
            ax1.arrow(x[k], y[k], math.cos(yaw[k])/map.xy_resolution/2, math.sin(yaw[k])/map.xy_resolution/2, color='c')
            plt.pause(0.001)
    ax1.set_title("Hybrid A* with RS curve")

    # Figure 2: Draw Start, Goal Location Map and Path
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.arrow(np.round(START[0]/map.xy_resolution), np.round(START[1]/map.xy_resolution), math.cos(START[2])/map.xy_resolution/2, math.sin(START[2])/map.xy_resolution/2, width=0.02/map.xy_resolution, color='g', label='Start')
    ax2.arrow(np.round(GOAL[0]/map.xy_resolution), np.round(GOAL[1]/map.xy_resolution), math.cos(GOAL[2])/map.xy_resolution/2, math.sin(GOAL[2])/map.xy_resolution/2, width=0.02/map.xy_resolution, color='b', label='Goal')
    ax2.set_xlim(0, map.img_size) 
    ax2.xaxis.set_major_formatter(FuncFormatter(scale))
    ax2.set_ylim(0, map.img_size)
    ax2.yaxis.set_major_formatter(FuncFormatter(scale))
    ax2.plot(map.obstacle_x, map.obstacle_y, "sk", label='Obstacles')
    ax2.plot(x, y, linewidth=2, color='r', zorder=0, label='Path')
    
    for node in closed_list_nodes:
        ax2.arrow(np.round(node.x/map.xy_resolution), np.round(node.y/map.xy_resolution),
                  math.cos(node.yaw)/map.xy_resolution/4, math.sin(node.yaw)/map.xy_resolution/4,
                  color='green', width=0.0001/map.xy_resolution, zorder=0)
        
    ax2.set_title("Hybrid A* with RS curve")
    ax2.legend()
    plt.show()

    # Figure 3: Draw Path, Map and Car Footprint
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(x, y, linewidth=1.5, color='r', zorder=0, label='Path')
    ax3.plot(map.obstacle_x, map.obstacle_y, "sk", label='Obstacles')
    ax3.set_xlim(0, map.img_size) 
    ax3.xaxis.set_major_formatter(FuncFormatter(scale))
    ax3.set_ylim(0, map.img_size)
    ax3.yaxis.set_major_formatter(FuncFormatter(scale))
    for k in np.arange(0, len(x), 2):
        draw_car(ax3, x[k], y[k], yaw[k])
        ax3.arrow(x[k], y[k], math.cos(yaw[k])/map.xy_resolution/2, math.sin(yaw[k])/map.xy_resolution/2, width=0.02/map.xy_resolution, color='c')
    ax3.set_title("Hybrid A* with RS curve")
    ax3.legend()
    plt.show()

if __name__ == '__main__':
    # Run Hybrid A*
    path_list, closed_list_nodes = HybridAStar().run()
    print(closed_list_nodes[0])
    x, y, yaw = zip(*[(np.round(node.x / map.xy_resolution), np.round(node.y / map.xy_resolution), node.yaw) for node in path_list])
    plot(x, y, yaw, closed_list_nodes)
