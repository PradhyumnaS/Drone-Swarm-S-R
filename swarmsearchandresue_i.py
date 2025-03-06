import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

class SwarmSearchRescueEnv(gym.Env):
    def __init__(self, grid_size=10, num_drones=3, num_victims=10, num_obstacles=10, max_steps=300):
        super(SwarmSearchRescueEnv, self).__init__()
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.num_victims = num_victims
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps

        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([5] * num_drones)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 4), dtype=np.float32)

        # Initialize positions and states
        self.drone_positions = np.zeros((num_drones, 2), dtype=int)
        self.victim_positions = np.zeros((num_victims, 2), dtype=int)
        self.obstacle_positions = np.zeros((num_obstacles, 2), dtype=int)
        self.rescued_victims = np.zeros(num_victims, dtype=bool)
        self.searched_area = np.zeros((grid_size, grid_size), dtype=bool)
        self.elevation_map = np.random.randint(0, 3, (grid_size, grid_size))
        self.base_position = np.array([0, 0])
        self.battery_levels = np.full(num_drones, 100)
        self.exploration_weights = np.ones((grid_size, grid_size))
        self.battery_drain_rate = 0.1  # Reduced battery drain rate
        self.exploration_decay = 0.95   # How much to reduce exploration weight for visited areas
        self.steps = 0
        self.metrics = {
        'total_rescued': 0,
        'exploration_rate': 0,
        'battery_efficiency': 0,
        'time_to_rescue': [],
        'total_distance': np.zeros(num_drones)}
        self.reset()

    def reset(self):
        # Reset all dynamic variables
        self.drone_positions = np.array([self.base_position] * self.num_drones)
        self.victim_positions = np.array([self._get_random_position() for _ in range(self.num_victims)])
        self.obstacle_positions = np.array([self._get_random_position() for _ in range(self.num_obstacles)])
        self.rescued_victims.fill(False)
        self.searched_area.fill(False)
        self.battery_levels.fill(100)
        self.steps = 0
        return self._get_observation()

    def _get_random_position(self):
        # Generate unique positions avoiding overlaps
        while True:
            pos = np.random.randint(0, self.grid_size, size=2)
            if not any(np.array_equal(pos, p) for p in np.vstack((self.drone_positions, self.victim_positions, self.obstacle_positions))):
                return pos

    def _get_observation(self):
        # Create grid observation with normalized values
        grid = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)
        grid[tuple(zip(*self.drone_positions)) + (0,)] = 1  # Drones
        for i, (x, y) in enumerate(self.victim_positions):
            if not self.rescued_victims[i]:
                grid[x, y, 1] = 1  # Victims
        grid[tuple(zip(*self.obstacle_positions)) + (2,)] = 1  # Obstacles
        grid[:, :, 3] = self.elevation_map / 2  # Elevation
        return grid

    def step(self, actions):
        reward = 0
        done = False
        self.steps += 1

        for i, action in enumerate(actions):
            if self.battery_levels[i] <= 0:
                continue

            # Store old position for distance calculation
            old_pos = self.drone_positions[i].copy()

            # Calculate new position
            new_pos = self.drone_positions[i].copy()
            if action == 0: new_pos[0] = max(0, new_pos[0]-1)
            elif action == 1: new_pos[0] = min(self.grid_size-1, new_pos[0]+1)
            elif action == 2: new_pos[1] = max(0, new_pos[1]-1)
            elif action == 3: new_pos[1] = min(self.grid_size-1, new_pos[1]+1)

            # Collision and movement handling
            if any(np.array_equal(new_pos, p) for p in self.obstacle_positions):
                reward -= 1
            else:
                # Update exploration weights and reward
                exploration_value = self.exploration_weights[new_pos[0], new_pos[1]]
                if not self.searched_area[new_pos[0], new_pos[1]]:
                    reward += 2 * exploration_value  # Higher reward for unexplored areas
                elif exploration_value > 0.3 and self.steps % 50 == 0:  # Periodic revisit
                    reward += 0.1 * exploration_value  # Small reward for strategic revisits
                else:
                    reward -= 0.1  # Small penalty for unnecessary revisits

                self.drone_positions[i] = new_pos
                self.searched_area[new_pos[0], new_pos[1]] = True
                self.exploration_weights[new_pos[0], new_pos[1]] *= self.exploration_decay

                # Update metrics
                self.metrics['total_distance'][i] += np.linalg.norm(new_pos - old_pos)

            # Battery management with slower drain
            self.battery_levels[i] -= self.battery_drain_rate
            if self.battery_levels[i] <= 10:
                dx = self.base_position[0] - self.drone_positions[i][0]
                dy = self.base_position[1] - self.drone_positions[i][1]
                self.drone_positions[i] += np.clip([dx, dy], -1, 1)
                if np.array_equal(self.drone_positions[i], self.base_position):
                    self.battery_levels[i] = 100
                    reward += 5

        # Victim rescue logic
        for i, victim in enumerate(self.victim_positions):
            if not self.rescued_victims[i] and any(np.array_equal(victim, p) for p in self.drone_positions):
                self.rescued_victims[i] = True
                reward += 30  # Balanced rescue reward

        # Termination conditions
        done = all(self.rescued_victims) or (self.steps >= self.max_steps) or all(self.battery_levels <= 0)
        if all(self.rescued_victims):
            reward += 100  # Mission success bonus

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        plt.clf()
        fig = plt.figure(figsize=(8, 4))  # Reduced figure size

        # Create main plot
        ax = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)  # Adjusted grid

        # Plot terrain using built-in terrain colormap
        terrain_plot = ax.imshow(self.elevation_map, cmap='terrain', alpha=0.7)

        # Add contour lines for elevation
        contour = ax.contour(self.elevation_map, colors='black', alpha=0.3, levels=3)

        # Plot explored area
        exploration = ax.imshow(self.searched_area.astype(float),
                              cmap='Blues', alpha=0.2)

        # Plot exploration weights
        heatmap = ax.imshow(self.exploration_weights, cmap='Reds', alpha=0.1)

        # Plot victims
        for i, victim_pos in enumerate(self.victim_positions):
            color = 'limegreen' if self.rescued_victims[i] else 'red'
            ax.plot(victim_pos[1], victim_pos[0], 'o', color=color,
                    markersize=10, markeredgecolor='white', markeredgewidth=1)

        # Plot obstacles
        for obs_pos in self.obstacle_positions:
            ax.plot(obs_pos[1], obs_pos[0], 's', color='black',
                    markersize=8, markeredgecolor='white', markeredgewidth=1)

        # Plot drones
        for i, (drone_pos, battery) in enumerate(zip(self.drone_positions, self.battery_levels)):
            # Drone marker
            ax.plot(drone_pos[1], drone_pos[0], '^', color='blue',
                    markersize=10, markeredgecolor='white', markeredgewidth=1)

            # Sensor range
            circle = plt.Circle((drone_pos[1], drone_pos[0]), 2,
                              color='blue', fill=False, alpha=0.3)
            ax.add_artist(circle)

            # Battery indicator with smaller font
            battery_color = 'green' if battery > 50 else 'red'
            ax.text(drone_pos[1], drone_pos[0]-0.3, f'{battery:.0f}%',
                    color=battery_color, ha='center', va='top',
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Base station
        ax.plot(self.base_position[1], self.base_position[0], '*',
                color='gold', markersize=15, markeredgecolor='orange',
                markeredgewidth=1, label='Base')

        # Add grid with reduced alpha
        ax.grid(True, alpha=0.2, linestyle=':')

        # Metrics subplot (made more compact)
        metrics_ax = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
        metrics_ax.axis('off')
        metrics_text = (
            f"Status\n"
            f"-------\n"
            f"Step: {self.steps}\n"
            f"Rescued: {sum(self.rescued_victims)}/{self.num_victims}\n"
            f"Explored: {(np.sum(self.searched_area)/self.grid_size**2)*100:.1f}%\n"
            f"Avg Batt: {np.mean(self.battery_levels):.1f}%"
        )
        metrics_ax.text(0, 0.95, metrics_text, transform=metrics_ax.transAxes,
                      verticalalignment='top', fontsize=8)

        plt.tight_layout()
        plt.pause(0.1)
        plt.close()

def evaluate_model(model_path, num_episodes=10, max_steps=1000):
    # Load the model
    model = PPO.load(model_path)
    env = SwarmSearchRescueEnv(grid_size=10, max_steps=max_steps)

    # Metrics to track
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'victims_rescued': [],
        'exploration_coverage': [],
        'battery_efficiency': [],
        'time_to_rescue': [],
        'success_rate': 0
    }

    print("\n=== Starting Model Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Maximum Steps: {max_steps}")
    print("================================")

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        rescue_times = []
        start_time = time.time()

        print(f"\nRunning Episode {episode + 1}/{num_episodes}")

        while steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            # Track when each victim is rescued
            current_rescued = sum(env.rescued_victims)
            if len(rescue_times) < current_rescued:
                rescue_times.append(steps)

            env.render()  # Visualize the episode

            if done:
                break

        # Calculate episode metrics
        exploration_rate = (np.sum(env.searched_area) / env.grid_size**2) * 100
        avg_battery = np.mean(env.battery_levels)
        victims_rescued = sum(env.rescued_victims)

        # Store metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(steps)
        metrics['victims_rescued'].append(victims_rescued)
        metrics['exploration_coverage'].append(exploration_rate)
        metrics['battery_efficiency'].append(avg_battery)
        metrics['time_to_rescue'].extend(rescue_times)

        # Print episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"Steps: {steps}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Victims Rescued: {victims_rescued}/{env.num_victims}")
        print(f"Exploration: {exploration_rate:.1f}%")
        print(f"Average Battery: {avg_battery:.1f}%")
        print(f"Time taken: {time.time() - start_time:.2f}s")

    # Calculate final metrics
    metrics['success_rate'] = sum(r == env.num_victims for r in metrics['victims_rescued']) / num_episodes * 100

    # Generate comprehensive report
    print("\n====== Final Evaluation Report ======")
    print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User: PradhyumnaS")
    print("===================================")
    print(f"Total Episodes: {num_episodes}")
    print(f"Success Rate: {metrics['success_rate']:.1f}%")
    print(f"Average Episode Length: {np.mean(metrics['episode_lengths']):.1f} steps")
    print(f"Average Reward: {np.mean(metrics['episode_rewards']):.2f}")
    print(f"Average Victims Rescued: {np.mean(metrics['victims_rescued']):.2f}")
    print(f"Average Exploration: {np.mean(metrics['exploration_coverage']):.1f}%")
    print(f"Average Battery Remaining: {np.mean(metrics['battery_efficiency']):.1f}%")
    print(f"Average Time per Rescue: {np.mean(metrics['time_to_rescue']):.1f} steps")
    print("\nPerformance Distribution:")
    print(f"Reward Range: [{min(metrics['episode_rewards']):.1f}, {max(metrics['episode_rewards']):.1f}]")
    print(f"Steps Range: [{min(metrics['episode_lengths'])}, {max(metrics['episode_lengths'])}]")
    print("===================================")

    return metrics

def plot_evaluation_results(metrics):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Plot rewards
    ax1.plot(metrics['episode_rewards'])
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')

    # Plot exploration coverage
    ax2.plot(metrics['exploration_coverage'])
    ax2.set_title('Exploration Coverage')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Coverage (%)')

    # Plot victims rescued
    ax3.plot(metrics['victims_rescued'])
    ax3.set_title('Victims Rescued')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Victims')

    # Plot battery efficiency
    ax4.plot(metrics['battery_efficiency'])
    ax4.set_title('Battery Efficiency')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Battery (%)')

    plt.tight_layout()
    plt.show()

model_path = "/content/drive/My Drive/models/Swarm Drones/swarmdrones.zip"  # Update with your model path
metrics = evaluate_model(model_path, num_episodes=5)  # Test for 5 episodes

plot_evaluation_results(metrics)