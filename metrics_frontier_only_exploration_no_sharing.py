import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import csv

# ============================
# Parameters (Adjustable)
# ============================
NUM_ROBOTS_LIST = range(15, 16)  # Number of robots from 9 to 13
MAP_WIDTH = 120           # Width of the environment map
MAP_HEIGHT = 120          # Height of the environment map
MAX_SPEED = 1.0           # Maximum speed of the robots
MAX_ACCELERATION = 0.5    # Maximum acceleration of the robots
SENSING_RANGE = 4         # Sensing range (R_s)
NUM_RUNS = 100             # Number of simulation runs per number of robots
MAX_FRAMES = 3000         # Maximum number of frames per run
VISUALIZE = False         # Set to True to enable visualization

metrics = []

# ============================
# Robot Class Definition
# ============================
class Robot:
    def __init__(self, id, x, y):
        self.id = id
        self.p = np.array([x, y], dtype=float)
        self.v = np.zeros(2, dtype=float)
        self.a = np.zeros(2, dtype=float)
        self.local_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool)
        self.f = None  # Current frontier target
        self.R_s = SENSING_RANGE        # Sensing range
        self.max_speed = MAX_SPEED
        self.max_acceleration = MAX_ACCELERATION
        # Metrics tracking
        self.total_distance = 0.0
        self.redundant_cells = 0

    def update_position(self, dt):
        # Update position towards the current frontier
        if self.f is None:
            self.update_frontier()
            if self.f is None:
                self.a = np.zeros(2)
                self.v = np.zeros(2)
                return

        direction = self.f - self.p
        distance = np.linalg.norm(direction)

        if distance > 1:
            desired_velocity = direction / distance * self.max_speed
            self.a = (desired_velocity - self.v) / dt
            self.a = np.clip(self.a, -self.max_acceleration, self.max_acceleration)
        else:
            self.f = None
            self.a = np.zeros(2)
            self.v = np.zeros(2)

        # Apply acceleration and update position and velocity
        self.v += self.a * dt
        self.v = np.clip(self.v, -self.max_speed, self.max_speed)
        self.p += self.v * dt

        # Keep robot within map bounds
        self.p = np.clip(self.p, [0, 0], [MAP_WIDTH - 1, MAP_HEIGHT - 1])

        # Calculate distance traveled
        self.total_distance += np.linalg.norm(self.v * dt)

    def sense_area(self, global_map):
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
                    global_map[j, i] = True  # Update the global map as well
        return new_cells

    def update_frontier(self):
        y, x = np.nonzero(~self.local_map)
        frontiers = [(x[i], y[i]) for i in range(len(x)) if self.is_frontier(x[i], y[i])]
        if frontiers:
            self.f = np.array(min(frontiers, key=lambda f: np.linalg.norm(np.array(f) - self.p)), dtype=float)
        else:
            self.f = None

    def is_frontier(self, x, y):
        if self.local_map[y, x]:
            return False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= y + dy < self.local_map.shape[0] and 0 <= x + dx < self.local_map.shape[1]:
                    if self.local_map[y + dy, x + dx]:
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

        # Initialize robots at random positions
        self.robots = []
        for i in range(num_robots):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)
            self.robots.append(Robot(i, x, y))

        # Metrics tracking
        self.total_new_area = 0
        self.total_steps = 0
        self.total_distance = 0.0
        self.total_redundant_cells = 0

    def update(self, dt):
        self.total_steps += 1
        for robot in self.robots:
            robot.update_position(dt)
        for robot in self.robots:
            new_cells = robot.sense_area(self.global_map)
            self.total_new_area += new_cells
            self.total_redundant_cells += robot.redundant_cells
            robot.redundant_cells = 0  # Reset for the next step
            self.total_distance += robot.total_distance
            robot.total_distance = 0  # Reset for the next step

# ============================
# Simulation Run Function
# ============================
def run_simulation():
    # Open a CSV file to write metrics
    with open('simulation_metrics_Only_Fontier_M120_R100_15.csv', mode='w', newline='') as csv_file:
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
                    if coverage == 1:  # Stop if coverage reaches 99%
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

        # Draw robots and their ranges
        for robot in env.robots:
            color = '#45b7d1'  # Color for robots

            # Draw sensing range as a filled circle
            sense_circle = Circle((robot.p[0], robot.p[1]), robot.R_s, fill=True, color=color, alpha=0.3)
            ax.add_artist(sense_circle)

            # Draw robot
            ax.plot(robot.p[0], robot.p[1], 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)

            if robot.f is not None:
                ax.plot([robot.p[0], robot.f[0]], [robot.p[1], robot.f[1]], '--', color=color, alpha=0.7, linewidth=1.5)

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Robot', markerfacecolor='#45b7d1', markersize=10),
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
    np.random.seed(25)
    env = Environment(MAP_WIDTH, MAP_HEIGHT, NUM_ROBOTS_LIST[0])  # Use the first number of robots for visualization

    anim = FuncAnimation(fig, update, frames=MAX_FRAMES, init_func=init, blit=True, interval=50)
    plt.show()
else:
    # Run the simulation without visualization
    run_simulation()
