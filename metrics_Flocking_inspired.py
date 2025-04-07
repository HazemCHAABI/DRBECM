import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import csv

# ============================
# Parameters (Adjustable)
# ============================
NUM_ROBOTS_LIST = range(5, 15)  # Number of robots from 9 to 13
MAP_WIDTH = 120           # Width of the environment map
MAP_HEIGHT = 120          # Height of the environment map
MAX_SPEED = 1.0           # Maximum speed of the robots
MAX_ACCELERATION = 0.5    # Maximum acceleration of the robots
COMMUNICATION_RANGE = 20  # Communication range (R_c)
SENSING_RANGE = 4         # Sensing range (R_s)
NUM_RUNS = 30             # Number of simulation runs per number of robots
MAX_FRAMES = 3000         # Maximum number of frames per run
VISUALIZE = True         # Set to True to enable visualization
STAGNATION_RADIUS = 5    # Radius to check for stagnation
STAGNATION_STEPS = 20     # Number of steps before considering robot as stagnant

# ============================
# Robot Class Definition
# ============================

metrics = []

class Robot:
    def __init__(self, id, x, y, role):
        self.id = id
        self.p = np.array([x, y], dtype=float)
        self.v = np.zeros(2, dtype=float)
        self.a = np.zeros(2, dtype=float)
        self.role = role
        self.local_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool)
        self.f = None  # Current frontier target
        self.R_c = COMMUNICATION_RANGE  # Communication range
        self.R_s = SENSING_RANGE        # Sensing range
        self.max_speed = MAX_SPEED
        self.max_acceleration = MAX_ACCELERATION
        self.connected_to = None  # Store the robot this one is connected to
        self.test = False
        self.rng_neighbors = []  # Store RNG neighbors
        # Metrics tracking for failures
        self.failure_count = 0
        self.total_attempts = 0
        # New attributes for stagnation detection
        self.position_history = []  # Store recent positions
        self.returning_to_supporter = False
        self.target_supporter = None
        # New attributes for metrics
        self.total_distance = 0.0
        self.redundant_cells = 0
        self.previous_position = np.copy(self.p)

    def update_role(self, robots):
        if self.role == 'base':
            return

        connected_robots = [r for r in robots if np.linalg.norm(self.p - r.p) <= self.R_c and r.id != self.id]
        base_station = next(r for r in robots if r.role == 'base')
        supporters = [r for r in connected_robots if r.role == 'supporter' and r.id != self.id]
        explorers = [r for r in connected_robots if r.role == 'explorer' and r.id != self.id]
        
        if self.role == 'explorer' and explorers: 
            if all(np.linalg.norm(self.p - r.p) > 0.75 * self.R_c for r in supporters + [base_station]) and self.test:
                self.role = 'supporter'
                self.f = None
                return

        if self.is_connected(robots) and self.role == 'supporter' and not explorers and len(supporters) == 1:
            if all(r.is_connected(robots) for r in connected_robots) and \
               any(np.linalg.norm(r.p - robots[0].p) < np.linalg.norm(self.p - robots[0].p) for r in connected_robots if r.role == 'supporter'):
                self.role = 'explorer'
                self.f = None
        
        if self.role == 'supporter' and not explorers and len(supporters) == 1 and not "base" in [r.role for r in connected_robots]:
            self.role = 'explorer'
            self.f = None

    def is_connected(self, robots):
        if self.role not in ['supporter', 'base']:
            return True  # Explorers do not need to be connected in this context
        visited = set()
        self._dfs_connect(robots, visited)
        return any(r.role == 'base' for r in visited)

    def _dfs_connect(self, robots, visited):
        visited.add(self)
        for r in robots:
            if r not in visited and np.linalg.norm(self.p - r.p) <= self.R_c and r.role in ['supporter', 'base']:
                r._dfs_connect(robots, visited)

    def check_stagnation(self):
        # Add current position to history
        self.position_history.append(np.copy(self.p))
        
        # Keep only the last STAGNATION_STEPS positions
        if len(self.position_history) > STAGNATION_STEPS:
            self.position_history.pop(0)
        
        # Check if we have enough history to determine stagnation
        if len(self.position_history) == STAGNATION_STEPS:
            # Calculate the maximum distance from the current position to any position in history
            max_distance = max(np.linalg.norm(self.p - pos) for pos in self.position_history[:-1])
            return max_distance <= STAGNATION_RADIUS
        return False

    def find_closest_supporter(self, robots):
        supporters = [r for r in robots if r.role in ['supporter', 'base']]
        if supporters:
            return min(supporters, key=lambda r: np.linalg.norm(self.p - r.p))
        return None

    def update_position(self, robots, dt):
        if self.role == 'base':
            return
        self.total_attempts += 1  

        # Store previous position to calculate distance traveled
        prev_p = np.copy(self.p)

        # Check for stagnation if robot is an explorer
        if self.role == 'explorer' and not self.returning_to_supporter:
            if self.check_stagnation():
                self.returning_to_supporter = True
                self.target_supporter = self.find_closest_supporter(robots)
                self.f = None  # Clear current frontier target

        if self.returning_to_supporter and self.target_supporter:
            # Move towards the closest supporter
            direction = self.target_supporter.p - self.p
            distance = np.linalg.norm(direction)
            
            if distance > self.R_s:  # Keep moving until within sensing range
                direction = direction / distance
                desired_velocity = direction * self.max_speed
                self.a = (desired_velocity - self.v) / dt
                self.a = np.clip(self.a, -self.max_acceleration, self.max_acceleration)
            else:
                # Reset stagnation detection when reached supporter
                self.returning_to_supporter = False
                self.target_supporter = None
                self.position_history.clear()
                self.f = None
        else:
            # Normal movement behavior
            if self.role == 'supporter':
                self.update_supporter_position(robots, dt)
            elif self.role == 'explorer':
                self.update_explorer_position(robots, dt)

        # Apply repulsion force from RNG neighbors
        min_distance = 5.0
        repulsion_force = np.zeros(2, dtype=float)

        for neighbor in self.rng_neighbors:
            if self.role == 'supporter' :
                distance = np.linalg.norm(self.p - neighbor.p)
                if 0 < distance < min_distance:
                    direction_away = (self.p - neighbor.p) / distance
                    repulsion_strength = (min_distance - distance) / min_distance
                    repulsion_force += direction_away * repulsion_strength
                elif distance == 0:
                    repulsion_force += np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

        self.a += repulsion_force

        # Update position and velocity using acceleration
        proposed_v = self.v + self.a * dt
        proposed_v = np.clip(proposed_v, -self.max_speed, self.max_speed)
        proposed_p = self.p + proposed_v * dt

        if self.role == 'supporter':
            if self.maintains_global_connectivity(proposed_p, robots):
                self.v = proposed_v
                self.p = proposed_p
            else:
                self.v = np.zeros(2)
                self.a = np.zeros(2)
                self.failure_count += 1
        else:
            self.v = proposed_v
            self.p = proposed_p

        # Calculate distance traveled and accumulate
        distance_traveled = np.linalg.norm(self.p - prev_p)
        self.total_distance += distance_traveled

    def update_supporter_position(self, robots, dt):
        base = next(r for r in robots if r.role == 'base')
        connected_robots = [r for r in robots if np.linalg.norm(self.p - r.p) <= self.R_c and r.id != self.id]
        explorers = [r for r in connected_robots if r.role == 'explorer' and r.id != self.id]
        supporters = [r for r in connected_robots if r.role == 'supporter' and r.id != self.id]

        if not connected_robots:
            target = base.p
        else:
            if explorers:
                # Move towards the mean position of connected explorers
                p_connected = np.mean([r.p for r in explorers], axis=0)
                target = p_connected
            else:
                # If no explorers, stay connected with other supporters and base
                s_connected = np.mean([r.p for r in supporters], axis=0) if supporters else base.p
                target = s_connected
        
        direction = target - self.p
        if np.linalg.norm(direction) > 0:
            desired_velocity = direction * self.max_speed * 0.7
        else:
            desired_velocity = np.zeros(2)
        self.a = (desired_velocity - self.v) / dt
        self.a = np.clip(self.a, -self.max_acceleration, self.max_acceleration)

    def update_explorer_position(self, robots, dt):
        base_station = next(r for r in robots if r.role == 'base')

        if not self.connected_to:
            self.f = base_station.p

        elif self.f is None:
            self.update_frontier(robots)
            if self.f is None:
                self.a = np.zeros(2)
                return
        
        direction = self.f - self.p
        distance = np.linalg.norm(direction)

        if distance > 1:
            desired_velocity = direction / distance * self.max_speed
            self.a = (desired_velocity - self.v) / dt
            self.a = np.clip(self.a, -self.max_acceleration, self.max_acceleration)
            
            # Check if moving in this direction will keep the explorer within communication range of supporters or base
            proposed_v = self.v + self.a * dt
            proposed_p = self.p + proposed_v * dt
            
            # Check if proposed_p is within communication range of at least one supporter or base
            connected = any(
                np.linalg.norm(proposed_p - r.p) <= self.R_c and r.role in ['supporter', 'base']
                for r in robots if r.id != self.id
            )
            if not connected:
                # Adjust the acceleration to keep within range
                # Find the closest supporter or base
                targets = [r for r in robots if r.role in ['supporter', 'base']]
                if targets:
                    closest_target = min(targets, key=lambda r: np.linalg.norm(self.p - r.p))
                    # Adjust direction towards the closest target
                    direction_to_target = closest_target.p - self.p
                    # Blend directions to move towards both the frontier and the supporter/base
                    direction = 0.7 * direction + 0.3 * direction_to_target
                    direction /= np.linalg.norm(direction)
                    desired_velocity = direction * self.max_speed
                    self.a = (desired_velocity - self.v) / dt
                    self.a = np.clip(self.a, -self.max_acceleration, self.max_acceleration)
        else:
            self.f = None
            self.a = np.zeros(2)
            self.v = np.zeros(2)

    def maintains_global_connectivity(self, proposed_position, robots):
        # Create a copy of the robots list with the proposed position
        temp_robots = [Robot(r.id, r.p[0], r.p[1], r.role) for r in robots]
        temp_self = next(r for r in temp_robots if r.id == self.id)
        temp_self.p = proposed_position

        # Check if all supporters and the base are still connected
        base = next(r for r in temp_robots if r.role == 'base')
        connected = set()
        base._dfs_connect_check(temp_robots, connected)
        supporters_and_base = [r for r in temp_robots if r.role in ['supporter', 'base']]
        return all(r in connected for r in supporters_and_base)

    def _dfs_connect_check(self, robots, connected):
        connected.add(self)
        for r in robots:
            if r not in connected and r.role in ['supporter', 'base']:
                if np.linalg.norm(self.p - r.p) <= 0.95 * self.R_c:
                    r._dfs_connect_check(robots, connected)

    def sense_area(self, global_map):
        if not (np.isfinite(self.p).all()):
            print(f"Warning: Invalid position detected for robot {self.id}: {self.p}")
            return 0
        x, y = np.clip(np.round(self.p), 0, np.array(global_map.shape) - 1).astype(int)

        new_cells = 0
        i_range = np.arange(max(0, x - self.R_s), min(global_map.shape[1], x + self.R_s + 1))
        j_range = np.arange(max(0, y - self.R_s), min(global_map.shape[0], y + self.R_s + 1))
        
        for i in i_range:
            for j in j_range:
                if (i - x)**2 + (j - y)**2 <= self.R_s**2:
                    if not self.local_map[j, i]:
                        new_cells += 1
                    else:
                        self.redundant_cells += 1  # Count redundant exploration
                    self.local_map[j, i] = True
                    global_map[j, i] = True
        return new_cells

    def share_information(self, robots):
        # Get RNG neighbors within communication range
        rng_neighbors = self.get_rng_neighbors(robots)
        self.rng_neighbors = rng_neighbors  # Store for visualization
        for r in rng_neighbors:
            r.local_map |= self.local_map

    def get_rng_neighbors(self, robots):
        rng_neighbors = []
        for r in robots:
            if r != self and np.linalg.norm(self.p - r.p) <= self.R_c:
                edge_exists = True
                d_self_r = np.linalg.norm(self.p - r.p)
                for c in robots:
                    if c != self and c != r:
                        if np.linalg.norm(self.p - c.p) <= self.R_c and np.linalg.norm(r.p - c.p) <= self.R_c:
                            if np.linalg.norm(self.p - c.p) < d_self_r and np.linalg.norm(r.p - c.p) < d_self_r:
                                edge_exists = False
                                break
                if edge_exists:
                    rng_neighbors.append(r)
        return rng_neighbors

    def update_frontier(self, robots):
        connected_robot = self.find_connected_robots(robots)
        self.connected_to = connected_robot
        self.test = False
        
        if self.role != 'explorer':
            self.f = None
            return

        if not connected_robot:
            self.f = None
            return

        y, x = np.nonzero(~self.local_map)
        frontiers = [(x[i], y[i]) for i in range(len(x)) if self.is_frontier(x[i], y[i])]
        valid_frontiers = []
        for robot in connected_robot:
            valid_frontiers += [f for f in frontiers if np.linalg.norm(np.array(f) - robot.p) <= self.R_c]
        
        if valid_frontiers:
            self.f = np.array(min(valid_frontiers, key=lambda f: np.linalg.norm(np.array(f) - self.p)), dtype=float)
            return
        
        elif frontiers:
            self.f = np.array(min(frontiers, key=lambda f: np.linalg.norm(np.array(f) - self.p)), dtype=float)
            self.test = True
            return
        
        if not valid_frontiers:
            valid_frontiers = None
            return

    def find_connected_robots(self, robots):
        L=[]
        if self.role == 'explorer':
            for robot in robots:
                if robot.role in ['supporter', 'base'] and np.linalg.norm(self.p - robot.p) <= self.R_c:
                    L.append(robot)
            if L:
                return L
            else:
                return None
        L=[]
        if self.role == 'supporter':
            for robot in robots:
                if robot.role in ['explorer', 'base', 'supporter'] and np.linalg.norm(self.p - robot.p) <= self.R_c:
                    L.append(robot)
            if L:
                return L
            else:
                return None

    def is_frontier(self, x, y):
        if self.local_map[y, x]:
            return False

        # Define the boundaries of the 3x3 neighborhood
        y_min = max(0, y - 1)
        y_max = min(self.local_map.shape[0], y + 2)  # +2 because slicing is exclusive at the end
        x_min = max(0, x - 1)
        x_max = min(self.local_map.shape[1], x + 2)

        # Extract the 3x3 neighborhood
        neighborhood = self.local_map[y_min:y_max, x_min:x_max]

        # Create a mask to exclude the center cell
        center_y = y - y_min
        center_x = x - x_min
        mask = np.ones(neighborhood.shape, dtype=bool)
        mask[center_y, center_x] = False

        # Check if any neighboring cells are explored
        if np.any(neighborhood[mask]):
            return True
        return False


# ============================
# Environment Class Definition
# ============================
class Environment:
    def __init__(self, width, height, num_robots):
        self.width = width
        self.height = height
        self.global_map = np.zeros((height, width), dtype=bool)
        
        # Place base station at the corner
        self.robots = [Robot(0, 1, 1, 'base')]
        
        # Place other robots randomly, but ensure they're connected to the base
        for i in range(1, num_robots):
            while True:
                x = np.random.uniform(2, 10)
                y = np.random.uniform(2, 10)
                new_robot = Robot(i, x, y, 'explorer')
                if new_robot.is_connected([self.robots[0]]):
                    self.robots.append(new_robot)
                    break

        # Metrics tracking
        self.total_new_area = 0
        self.total_steps = 0
        self.failure_count = 0
        self.total_distance = 0.0
        self.total_redundant_cells = 0

    def update(self, dt):
        self.total_steps += 1
        for robot in self.robots:
            robot.update_role(self.robots)
        for robot in self.robots:
            robot.previous_position = robot.p.copy()
            robot.update_position(self.robots, dt)
        for robot in self.robots:
            new_cells = robot.sense_area(self.global_map)
            self.total_new_area += new_cells
            if not all(robot.previous_position == robot.p):
                self.total_redundant_cells += robot.redundant_cells
            robot.redundant_cells = 0  # Reset for the next step
            self.failure_count += robot.failure_count
            robot.failure_count = 0  # Reset for the next step
            self.total_distance += robot.total_distance
            robot.total_distance = 0  # Reset for the next step
        for robot in self.robots:
            robot.share_information(self.robots)
        for robot in self.robots:
            robot.update_frontier(self.robots)

# ============================
# Simulation Run Function
# ============================
def run_simulation():
    # Open a CSV file to write metrics
    with open('test1.csv', mode='w', newline='') as csv_file:
        fieldnames = ['num_robots', 'run', 'exploration_time', 'coverage', 'redundant_exploration', 'total_distance']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for num_robots in NUM_ROBOTS_LIST:
            print(f"Running simulations for NUM_ROBOTS = {num_robots}")
            for run in range(NUM_RUNS):
                print(f"  Run {run + 1}/{NUM_RUNS}")
                np.random.seed(run)  # Use run number as seed
                env = Environment(MAP_WIDTH, MAP_HEIGHT, num_robots)
                # Run the simulation for MAX_FRAMES or until coverage reaches 99%
                for frame in range(MAX_FRAMES):
                    env.update(dt=1)
                    coverage = env.global_map.sum() / (env.width * env.height)
                    if coverage >= 0.99:  # Stop if coverage reaches 99%
                        break
                # Collect metrics
                exploration_time = frame + 1  # Since frame starts from 0
                coverage = env.global_map.sum() / (env.width * env.height)
                redundant_exploration = env.total_redundant_cells
                total_distance = env.total_distance
                metrics.append({
                    'num_robots': num_robots,
                    'exploration_time': exploration_time,
                    'coverage': coverage,
                    'redundant_exploration': redundant_exploration,
                    'total_distance': total_distance
                })
                # Write metrics to CSV
                writer.writerow({
                    'num_robots': num_robots,
                    'run': run + 1,
                    'exploration_time': exploration_time,
                    'coverage': coverage,
                    'redundant_exploration': redundant_exploration,
                    'total_distance': total_distance
                })

    # Compute average metrics for each number of robots
    for num_robots in NUM_ROBOTS_LIST:
        metrics_subset = [m for m in metrics if m['num_robots'] == num_robots]
        avg_exploration_time = np.mean([m['exploration_time'] for m in metrics_subset])
        avg_coverage = np.mean([m['coverage'] for m in metrics_subset])
        avg_redundant_exploration = np.mean([m['redundant_exploration'] for m in metrics_subset])
        avg_total_distance = np.mean([m['total_distance'] for m in metrics_subset])

        print(f"\nAverage Metrics over {NUM_RUNS} runs for NUM_ROBOTS = {num_robots}:")
        print("Average Exploration Time: {:.2f} steps".format(avg_exploration_time))
        print("Average Coverage: {:.2%}".format(avg_coverage))
        print("Average Redundant Exploration: {:.2f} cells".format(avg_redundant_exploration))
        print("Average Total Distance Traveled: {:.2f} units".format(avg_total_distance))

# ============================
# Visualization (Optional)
# ============================
if VISUALIZE:
    # Visualization setup
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))

    def init():
        ax.clear()
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        return []

    def update(frame):
        env.update(dt=1)
        ax.clear()
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        
        # Create custom colormap for explored area
        colors = ['#1a1a1a', '#4a4a4a', '#7a7a7a', '#aaaaaa']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Draw explored area
        ax.imshow(env.global_map, cmap=cmap, alpha=0.7)
        
        # Draw RNG communication links
        drawn_edges = set()
        for robot in env.robots:
            for r in robot.rng_neighbors:
                edge = tuple(sorted((robot.id, r.id)))
                if edge not in drawn_edges:
                    ax.plot([robot.p[0], r.p[0]], [robot.p[1], r.p[1]], 
                            '--', color='red', alpha=0.5, linewidth=1)
                    drawn_edges.add(edge)
        
        # Draw communication links
        for robot in env.robots:
            if robot.connected_to:
                for r in robot.connected_to:
                    ax.plot([robot.p[0], r.p[0]], [robot.p[1], r.p[1]], 
                            '-', color='#4a4a4a', alpha=0.5, linewidth=0.8)
        
        # Draw robots and their ranges
        for robot in env.robots:
            color = '#ff6b6b' if robot.role == 'base' else '#fc03e8' if robot.role == 'supporter' else '#45b7d1'
            
            # Draw communication range
            comm_circle = Circle((robot.p[0], robot.p[1]), robot.R_c, fill=False, color=color, alpha=0.4)
            ax.add_artist(comm_circle)
            
            # Draw sensing range as a filled circle
            sense_circle = Circle((robot.p[0], robot.p[1]), robot.R_s, fill=True, color=color, alpha=0.3)
            ax.add_artist(sense_circle)
            
            # Draw robot
            ax.plot(robot.p[0], robot.p[1], 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
            
            if robot.f is not None:
                ax.plot([robot.p[0], robot.f[0]], [robot.p[1], robot.f[1]], '--', color=color, alpha=0.7, linewidth=1.5)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Base Station', markerfacecolor='#ff6b6b', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Supporter', markerfacecolor='#fc03e8', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Explorer', markerfacecolor='#45b7d1', markersize=10),
            plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='RNG Communication Link'),
            plt.Line2D([0], [0], color='#4a4a4a', lw=2, label='Communication Link'),
            plt.Line2D([0], [0], color='#45b7d1', lw=2, linestyle='--', label='Frontier Path')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize=10)
        
        # Set title and adjust layout
        explored_percentage = env.global_map.sum() / (env.width * env.height)
        ax.set_title(f'Robot Exploration Progress: {explored_percentage:.2%}', fontsize=16, pad=20)
        ax.set_facecolor('#1a1a1a')
        fig.tight_layout()
        
        return []

    # Initialize environment
    np.random.seed(14)
    env = Environment(MAP_WIDTH, MAP_HEIGHT, NUM_ROBOTS_LIST[0])  # Use the first number of robots for visualization

    anim = FuncAnimation(fig, update, frames=MAX_FRAMES, init_func=init, blit=True, interval=50)
    plt.show()
else:
    # Run the simulation without visualization
    run_simulation()