import sys
import numpy as np
import pygame
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from gym import Env
from gym.spaces import Discrete, Box
from pygame import Rect

# Constants from csettings.py
IDLE = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

# Graphics Constants    
WINDOW_PIXELS = 700
BG_COLOR = (255, 255, 255)
GRID_COLOR = (171, 171, 171)
AGENT_COLOR = (0, 0, 255)
RESOURCE_COLOR = (10, 155, 0)
RADIUS_COLOR = (180, 255, 180)
AGITATED_COLOR = (255, 0, 0)

# Keep the GridworldMultiAgentv25 class as it was in the original script
# ... [entire GridworldMultiAgentv25 class remains the same] ...
class GridworldMultiAgentv25(Env):
    def __init__(self, nb_agents=2, agent_power=1, nb_resources=2, nb_civilians=5, gridsize=5, radius=1, nb_steps=50,
                 reward_extracting=10.0, alpha=6, beta=0, reward_else=-1.0, seed=1, screen=None, debug=False):
        # Debug mode
        self.debug = debug

        # Activate graphics if specified
        if screen is not None:
            self.screen = screen
        # Compute cell pixel size
        self.cell_pixels = WINDOW_PIXELS / gridsize

        # Set number of possible actions and reset step number
        self.nb_actions = 5
        self.step_nb = 0

        # Set environment variables
        self.nb_agents = nb_agents
        self.agent_power = agent_power
        self.nb_resources = nb_resources
        self.nb_civilians = nb_civilians
        self.gridsize = gridsize
        self.radius = radius
        self.nb_steps = nb_steps
        self.reward_extracting = reward_extracting
        self.alpha = alpha
        self.beta = beta
        self.reward_else = reward_else

        # Set up action and observation spaces
        self.action_space = Discrete(self.nb_actions ** self.nb_agents)
        self.observation_space = Box(
            np.zeros(3 * self.nb_resources + 2 * self.nb_agents),
            np.ones(3 * self.nb_resources + 2 * self.nb_agents)
        )

        # Set random seed for testing
        np.random.seed(seed)
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))

        # Map action space to the base of possible actions
        self.action_map = {}
        for i in range(self.action_space.n):
            action = [0] * self.nb_agents
            num = i
            index = -1
            while num > 0:
                action[index] = num % self.nb_actions
                num = num // self.nb_actions
                index -= 1
            self.action_map[i] = action

    def step(self, action: int):
        for i, a in enumerate(self.action_map[action]):
            if a == UP:
                self.state_agent[i, 1] = max(0, self.state_agent[i, 1] - 1)
            elif a == RIGHT:
                self.state_agent[i, 0] = min(self.gridsize - 1, self.state_agent[i, 0] + 1)
            elif a == DOWN:
                self.state_agent[i, 1] = min(self.gridsize - 1, self.state_agent[i, 1] + 1)
            elif a == LEFT:
                self.state_agent[i, 0] = max(0, self.state_agent[i, 0] - 1)

        reward = self.reward_else
        extracted_resources = []
        for i, resource in enumerate(self.state_resources):
            nb_agents_radius = 0
            nb_civilians_radius = 0
            extracted = False
            for agent in self.state_agent:
                if (resource[0] - self.radius <= agent[0] <= resource[0] + self.radius and
                        resource[1] - self.radius <= agent[1] <= resource[1] + self.radius):
                    nb_agents_radius += 1
                    if np.all(resource == agent):
                        extracted = True
                        extracted_resources.append(i)
            if extracted:
                for civilian in self.state_civilians:
                    if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                            resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                        nb_civilians_radius += 1
                reward += self.reward_extracting
                riot_size = (nb_civilians_radius - self.agent_power * nb_agents_radius)
                if riot_size > 0:
                    reward -= self.alpha * riot_size
                else:
                    reward -= self.beta * riot_size

        for i in extracted_resources:
            self.state_resources[i, :] = np.random.randint(self.gridsize, size=2)

        self.step_nb += 1
        done = self.step_nb == self.nb_steps
        info = {}

        observation = self.observe()

        if self.debug:
            print("Reward:", reward)
            print('Observation:', observation)

        return observation, reward, done, info

    def reset(self):
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        self.state_resources = np.random.randint(self.gridsize, size=(self.nb_resources, 2))
        self.state_civilians = np.random.randint(self.gridsize, size=(self.nb_civilians, 2))
        self.step_nb = 0
        return self.observe()

    def render(self, mode='human'):
        # Render background
        self.screen.fill(BG_COLOR)

        # Draw resources and their radii as cell-sized green squares
        for resource in self.state_resources:
            for x in range(resource[0] - self.radius, resource[0] + self.radius + 1):
                for y in range(resource[1] - self.radius, resource[1] + self.radius + 1):
                    rect = Rect(int(x * self.cell_pixels), int(y * self.cell_pixels), 
                                int(self.cell_pixels), int(self.cell_pixels))
                    pygame.draw.rect(self.screen, RADIUS_COLOR, rect)

        for resource in self.state_resources:
            rect = Rect(int(resource[0] * self.cell_pixels), int(resource[1] * self.cell_pixels), 
                        int(self.cell_pixels), int(self.cell_pixels))
            pygame.draw.rect(self.screen, RESOURCE_COLOR, rect)

        # Draw cell grid as grey bars
        for i in range(self.gridsize + 1):
            pygame.draw.line(self.screen, GRID_COLOR, 
                             (int(i * self.cell_pixels), 0),
                             (int(i * self.cell_pixels), WINDOW_PIXELS - 1), 1)
            pygame.draw.line(self.screen, GRID_COLOR, 
                             (0, int(i * self.cell_pixels)),
                             (WINDOW_PIXELS - 1, int(i * self.cell_pixels)), 1)

        # Draw agents as blue circles
        for agent in self.state_agent:
            center = (
                int(agent[0] * self.cell_pixels + self.cell_pixels / 2), 
                int(agent[1] * self.cell_pixels + self.cell_pixels / 2)
            )
            pygame.draw.circle(self.screen, AGENT_COLOR, center, int(self.cell_pixels / 3))

        # Draw civilians as red circles
        for civilian in self.state_civilians:
            center = (
                int(civilian[0] * self.cell_pixels + self.cell_pixels / 2),
                int(civilian[1] * self.cell_pixels + self.cell_pixels / 2)
            )
            pygame.draw.circle(self.screen, AGITATED_COLOR, center, int(self.cell_pixels / 5))

        pygame.display.flip()

    def observe(self):
        norm_agents = self.state_agent.flatten().astype(float) / (self.gridsize - 1)
        norm_resources = []
        for resource in self.state_resources:
            nb_civilians_close = 0
            for civilian in self.state_civilians:
                if (resource[0] - self.radius <= civilian[0] <= resource[0] + self.radius and
                        resource[1] - self.radius <= civilian[1] <= resource[1] + self.radius):
                    nb_civilians_close += 1
            norm_resources += [resource[0] / (self.gridsize - 1), resource[1] / (self.gridsize - 1),
                               nb_civilians_close / self.nb_civilians]

        return np.concatenate((np.array(norm_resources), norm_agents))
    

def pygame_to_opencv(surface):
    """Convert a pygame surface to a numpy array for OpenCV"""
    surface_array = pygame.surfarray.array3d(surface)
    opencv_image = cv2.transpose(surface_array)
    opencv_image = cv2.flip(opencv_image, 0)
    return opencv_image

def plot_rewards(total_rewards):
    """
    Create multiple visualizations of the rewards
    
    Args:
    total_rewards (list): List of rewards from each episode
    """
    plt.figure(figsize=(15, 5))

    # 1. Line plot of rewards
    plt.subplot(131)
    plt.plot(total_rewards, marker='o')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # 2. Histogram of rewards
    plt.subplot(132)
    plt.hist(total_rewards, bins='auto', edgecolor='black')
    plt.title('Distribution of Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')

    # 3. Box plot of rewards
    plt.subplot(133)
    plt.boxplot(total_rewards)
    plt.title('Reward Box Plot')
    plt.ylabel('Reward')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('random_agent_rewards_analysis.png')
    plt.close()

def main():
    # Initialize pygame and screen
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_PIXELS, WINDOW_PIXELS))
    pygame.display.set_caption('Random Agent Resource Extraction Game')

    # Create environment 
    env = GridworldMultiAgentv25(gridsize=5, nb_agents=2, nb_resources=2, screen=screen, debug=False)

    # Video recording setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter('random_agent_output1.mp4', fourcc, 10, (WINDOW_PIXELS, WINDOW_PIXELS))

    # Reward tracking
    total_rewards = []
    
    # Total steps to run
    MAX_TOTAL_STEPS = 1000

    # Seed for reproducibility
    np.random.seed(42)

    # Initial environment reset
    obs = env.reset()

    # Run the simulation
    for step in range(MAX_TOTAL_STEPS):
        # Choose a random action from the action space
        action = np.random.randint(env.action_space.n)
        
        # Step the environment
        obs, reward, done, _ = env.step(action)
        
        # Record reward
        total_rewards.append(reward)
        
        # Render the environment
        env.render()
        
        # Capture the pygame screen
        screen_capture = pygame_to_opencv(pygame.display.get_surface())
        video_output.write(screen_capture)
        
        # Print step results
        print(f"Step {step+1} - Action: {action}, Reward: {reward}")
        
        # Small delay to control frame rate
        pygame.time.delay(100)
        
        # Reset if environment is done
        if done:
            obs = env.reset()

    # Close resources
    video_output.release()
    pygame.quit()

    # Print reward statistics
    print("\nReward Statistics:")
    print(f"Total Steps: {len(total_rewards)}")
    print(f"Average Reward: {np.mean(total_rewards)}")
    print(f"Cumulative Reward: {np.sum(total_rewards)}")
    print(f"Maximum Instantaneous Reward: {np.max(total_rewards)}")
    print(f"Minimum Instantaneous Reward: {np.min(total_rewards)}")

    # Save rewards to a CSV file
    np.savetxt('random_agent_rewards_log.csv', total_rewards, delimiter=',')

    # Create reward visualizations
    plot_rewards(total_rewards)

if __name__ == "__main__":
    main()