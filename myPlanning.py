import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial as kd
import reeds_shepp as rsCurve
from dataclasses import dataclass, field
import time

IMAGE_PATH = 'maze.jpg' 
THRESH = 127              
MAP_HIGHT = 200         
MAP_WIDTH = 200       
START =[100, 100, 0]
GOAL = [135, 125, 0]

class Car:
    max_steer_angle = 0.6
    steer_precision = 10
    wheel_base = 3.5
    axle2front = 4.5
    axle2back = 1
    width = 2
    reverse_cost = 30
    dir_change_cost = 30
    yaw_change_cost = 5
    motion_commands =[[i, dir] for i in np.arange(max_steer_angle, -(max_steer_angle + \
                    max_steer_angle/steer_precision), -max_steer_angle/steer_precision) for dir in [1, -1]]
    
class Map:
    xy_resolution = 1     # grid block length
    yaw_resolution = np.deg2rad(15.0)  # grid block possible yaws 
    def __init__(self, img_path, thresh, high, width): 
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)                     
        thresh, map_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY) 
        map_img = cv2.resize(map_img, (width, high))                               
        self.map_array = np.array(map_img)
        self.high = high
        self.width = width
        self.obstacle_x, self.obstacle_y = [], []
        self.generate_obstacles()
        # create a KDTree to represent obstacles
        self.ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(self.obstacle_x, self.obstacle_y)])

    def generate_obstacles(self):
        for i in range(self.width):
            for j in range(self.high):
                if self.map_array[j][i] == 0:
                    self.obstacle_x.append(i)
                    self.obstacle_y.append(j)
    
    @staticmethod
    def config_idx(*args):
        """Index configuration (x, y, yaw) as map grid node"""
        return (round(args[0] / Map.xy_resolution), round(args[1] / Map.xy_resolution), round(args[2] / Map.yaw_resolution))

map = Map(IMAGE_PATH, THRESH, MAP_HIGHT, MAP_WIDTH)

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
        self.x_idx, self.y_idx, self.yaw_idx= Map.config_idx(self.x, self.y, self.yaw)
        self.F = self.G + self.heuristic(GOAL)
    
    def next_node(self, command, step):
        # Iterate next node
        next_x, next_y, next_yaw = self.kinematic_simulate(command, step)
        angle, dir = command
        G = self.G + self.heuristic([next_x, next_y]) + self.hybrid_cost(command)
        return HybridNode(next_x, next_y, next_yaw, parent=self, dir=dir, angle=angle, G=G)
        
    def __eq__(self, other: "HybridNode"):
        return self.x_idx == other.x_idx and self.y_idx == other.y_idx and self.yaw_idx == other.yaw_idx
        
    def __le__(self, other: "HybridNode"):
        return self.F <= other.F
    
    def __lt__(self, other: "HybridNode"):
        return self.F < other.F
    
    def __hash__(self) -> int:
        return hash((self.x_idx, self.y_idx, self.yaw_idx))

    def is_end(self, err = 0.05):
        if self.F - self.G < err:
            return True
        return False

    def is_collided(self):
        # out of map?
        if 0 > self.x or self.x >= map.width or 0 > self.y or self.y >= map.high:
            return True
        
        # collides with obstacles?
        cos_ = math.cos(self.yaw)
        sin_ = math.sin(self.yaw)
        half_len = (Car.axle2front + Car.axle2back)/2 + 1
        dl = (Car.axle2front - Car.axle2back)/2
        cx = self.x + dl * cos_
        cy = self.y + dl * sin_
        pointsInObstacle = map.ObstacleKDTree.query_ball_point([cx, cy], half_len)
        if pointsInObstacle:
            for p in pointsInObstacle:
                xo = map.obstacle_x[p] - cx
                yo = map.obstacle_y[p] - cy
                dx = xo * cos_ + yo * sin_
                dy = -xo * sin_ + yo * cos_
                if abs(dx) < half_len and abs(dy) < Car.width / 2 + 1:
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
    
@dataclass
class SetQueue:
    queue: set[HybridNode] = field(default_factory=set)
    def __bool__(self):
        return bool(self.queue)
    
    def __contains__(self, item):
        return item in self.queue

    def __len__(self):
        return len(self.queue)
    
    def get(self):
        node = min(self.queue)  # O(n)?
        self.queue.remove(node) # O(1)
        return node
        
    def put(self, node: HybridNode):
        if node in self.queue:              # O(1)
            qlist = list(self.queue)       
            idx = qlist.index(node)         # O(n)
            if node.F < qlist[idx].F:
                self.queue.remove(node)     # O(1)
                self.queue.add(node)       
        else:
            self.queue.add(node)            # O(1)

    def empty(self):
        return len(self.queue) == 0
    
def reedsSheppNode(current_node, goal_node):
   # Instantaneous Radius of Curvature
    radius = math.tan(Car.max_steer_angle)/Car.wheel_base
    # Find all possible reeds-shepp paths between current and goal node
    reedsSheppPaths = rsCurve.calc_all_paths(current_node.x, current_node.y, current_node.yaw, \
                                             goal_node.x, goal_node.y, goal_node.yaw, radius, 1)
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
        # traj=[]
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
    def __init__(self, move_step=4, step=0.8):
        self.start = HybridNode(*START)
        self.step = step
        self.move_step = move_step
        self.close_set = set()                    
        self.open_queue = SetQueue()              
        self.path_list = []                       
        self.rSNode = None
        # Error Check
        self.end = HybridNode(*GOAL)
        if self.start.is_collided():
            raise ValueError(f"START on the obstacles!")
        if self.end.is_collided():
            raise ValueError(f"TARGET on the obstacles!")

    def _update_open_list(self, curr: HybridNode):
        for command in Car.motion_commands:
            next_node = curr
            for _ in range(self.move_step):
                next_node = next_node.next_node(command, self.step) 
            if next_node.is_collided():
                continue
            if next_node in self.close_set:
                continue
            
            self.open_queue.put(next_node)
    
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
        self.open_queue.put(self.start)
        # Start forward search
        self.tic()
        while not self.open_queue.empty():
            # Pop the node with the minimum F cost from OpenList
            curr: HybridNode = self.open_queue.get()
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
            # End iteration if the end is reached
            if curr.is_end():
                break
        print("Done.\n")
        self.toc()

        # Combine nodes to form the path
        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()
        return self.path_list

def draw_car(ax, x, y, yaw, color='black'):
    car = np.array([[-Car.axle2back, -Car.axle2back, Car.axle2front, Car.axle2front, -Car.axle2back],
                    [Car.width / 2, -Car.width / 2, -Car.width / 2, Car.width / 2, Car.width / 2]])

    rotation_z = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])
    car = np.dot(rotation_z, car)
    car += np.array([[x], [y]])
    ax.plot(car[0, :], car[1, :], color)

def plot(x, y, yaw):
    # Figure 1: Draw Animated Car
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for _ in range(2):
        for k in range(len(x)):
            ax1.cla()
            ax1.set_xlim(min(map.obstacle_x), max(map.obstacle_x)) 
            ax1.set_ylim(min(map.obstacle_y), max(map.obstacle_y))
            ax1.plot(map.obstacle_x, map.obstacle_y, "sk")
            ax1.plot(x, y, linewidth=1.5, color='r', zorder=0)
            draw_car(ax1, x[k], y[k], yaw[k])
            ax1.arrow(x[k], y[k], 1 * math.cos(yaw[k]), 1 * math.sin(yaw[k]), width=0.1, color='c')
            plt.pause(0.001)
    ax1.set_title("Hybrid A* with RS curve")

    # Figure 2: Draw Start, Goal Location Map and Path
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.arrow(START[0], START[1], math.cos(START[2]), math.sin(START[2]), width=0.1, color='g', label='Start')
    ax2.arrow(GOAL[0], GOAL[1], math.cos(GOAL[2]), math.sin(GOAL[2]), width=0.1, color='b', label='Goal')
    ax2.set_xlim(min(map.obstacle_x), max(map.obstacle_x)) 
    ax2.set_ylim(min(map.obstacle_y), max(map.obstacle_y))
    ax2.plot(map.obstacle_x, map.obstacle_y, "sk", label='Obstacles')
    ax2.plot(x, y, linewidth=2, color='r', zorder=0, label='Path')
    ax2.set_title("Hybrid A* with RS curve")
    ax2.legend()

    # Figure 3: Draw Path, Map and Car Footprint
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(x, y, linewidth=1.5, color='r', zorder=0, label='Path')
    ax3.plot(map.obstacle_x, map.obstacle_y, "sk", label='Obstacles')
    for k in np.arange(0, len(x), 2):
        ax3.set_xlim(0, map.width) 
        ax3.set_ylim(0, map.high)
        draw_car(ax3, x[k], y[k], yaw[k])
        ax3.arrow(x[k], y[k], 1 * math.cos(yaw[k]), 1 * math.sin(yaw[k]), width=0.1, color='c')
    ax3.set_title("Hybrid A* with RS curve")
    ax3.legend()
    plt.show()

if __name__ == '__main__':
    # Run Hybrid A*
    x, y, yaw = zip(*[(node.x, node.y, node.yaw) for node in HybridAStar().run()])
    plot(x, y, yaw) 