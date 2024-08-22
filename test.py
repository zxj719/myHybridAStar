import numpy as np
import heapq

class HybridAStar:
    def __init__(self, start, goal, obstacle_map, grid_resolution, theta_resolution):
        self.start = start  # (x, y, theta)
        self.goal = goal  # (x, y, theta)
        self.obstacle_map = obstacle_map
        self.grid_resolution = grid_resolution
        self.theta_resolution = theta_resolution
        self.open_list = []
        self.closed_list = set()
        self.goal_threshold = grid_resolution  # Threshold to consider goal reached

    def heuristic_non_holonomic(self, x, y, theta):
        # Heuristic 1: Non-holonomic without obstacles
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        dtheta = np.abs(self.goal[2] - theta)
        non_holonomic_cost = np.sqrt(dx**2 + dy**2) + dtheta
        euclidean_cost = np.sqrt(dx**2 + dy**2)
        return max(non_holonomic_cost, euclidean_cost)

    def heuristic_obstacle_based(self, x, y):
        # Heuristic 2: Dynamic programming with obstacles
        min_distance = np.inf
        distance = np.sqrt((self.goal[0] - x)**2 + (self.goal[1] - y)**2)
        if self.obstacle_map[int(x/self.grid_resolution), int(y/self.grid_resolution)] == 1:
            distance += np.inf  # Penalize if in obstacle
        min_distance = min(min_distance, distance)
        return min_distance

    def heuristic(self, x, y, theta):
        return max(self.heuristic_non_holonomic(x, y, theta), self.heuristic_obstacle_based(x, y))

    def is_goal(self, x, y, theta):
        return np.sqrt((self.goal[0] - x)**2 + (self.goal[1] - y)**2) < self.goal_threshold and abs(self.goal[2] - theta) < self.theta_resolution

    def reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    def run(self):
        # Initialize the start node
        start_node = (self.start[0], self.start[1], self.start[2], 0, self.heuristic(*self.start), None)
        heapq.heappush(self.open_list, start_node)
        
        while self.open_list:
            current_node = heapq.heappop(self.open_list)
            x, y, theta, cost, heuristic_cost, parent = current_node
            
            # Goal check
            if self.is_goal(x, y, theta):
                return self.reconstruct_path(current_node)
            
            # Mark the node as visited
            self.closed_list.add((x, y, theta))
            
            # Expand the node
            for motion in self.get_motion_model():
                new_x = x + motion[0]
                new_y = y + motion[1]
                new_theta = theta + motion[2]
                new_cost = cost + motion[3]
                
                if not self.is_valid(new_x, new_y, new_theta):
                    continue
                
                if (new_x, new_y, new_theta) in self.closed_list:
                    continue
                
                new_heuristic_cost = self.heuristic(new_x, new_y, new_theta)
                new_node = (new_x, new_y, new_theta, new_cost, new_heuristic_cost, current_node)
                heapq.heappush(self.open_list, new_node)
        
        return None  # No path found
    
    def is_valid(self, x, y, theta):
        # Check if the node is within bounds and not in an obstacle
        grid_x = int(x / self.grid_resolution)
        grid_y = int(y / self.grid_resolution)
        if grid_x < 0 or grid_y < 0 or grid_x >= self.obstacle_map.shape[0] or grid_y >= self.obstacle_map.shape[1]:
            return False
        return self.obstacle_map[grid_x, grid_y] == 0
    
    def get_motion_model(self):
        # Possible motion model (dx, dy, dtheta, cost)
        motion_model = [
            (self.grid_resolution, 0, 0, self.grid_resolution),
            (0, self.grid_resolution, 0, self.grid_resolution),
            (-self.grid_resolution, 0, 0, self.grid_resolution),
            (0, -self.grid_resolution, 0, self.grid_resolution),
            (self.grid_resolution, self.grid_resolution, np.pi/4, np.sqrt(2) * self.grid_resolution),
            (-self.grid_resolution, -self.grid_resolution, -np.pi/4, np.sqrt(2) * self.grid_resolution),
            (self.grid_resolution, -self.grid_resolution, np.pi/4, np.sqrt(2) * self.grid_resolution),
            (-self.grid_resolution, self.grid_resolution, -np.pi/4, np.sqrt(2) * self.grid_resolution),
        ]
        return motion_model

# Example usage
if __name__ == "__main__":
    start = (10, 10, 0)
    goal = (50, 50, 0)
    obstacle_map = np.zeros((100, 100))  # Assuming a 100x100 grid with no obstacles
    obstacle_map[20:30, 20:30] = 1  # Example obstacle
    grid_resolution = 1
    theta_resolution = np.pi / 18  # 10 degrees

    planner = HybridAStar(start, goal, obstacle_map, grid_resolution, theta_resolution)
    path = planner.run()
    if path:
        print("Path found!")
        for step in path:
            print(step)
    else:
        print("No path found.")
