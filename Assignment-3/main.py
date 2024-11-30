from env import MAPFEnvironment
from rollout import RolloutPlanner
import pygame
import time


def main():
    # Create environment and planner
    env = MAPFEnvironment(use_pygame=True)
    planner = RolloutPlanner(env, num_rollouts=50, depth=150)

    # Reset environment
    env.reset()
    total_steps = 0
    max_episodes = 100
    clock = pygame.time.Clock()

    running = True
    while running and total_steps < max_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get actions from rollout planner
        actions = planner.do_rollout()

        # Take step in environment
        next_states, reward, done, info = env.step(actions)
        total_steps += 1

        # Render current state
        env.render()
        print(f"Step {total_steps}, Reward: {reward}, Done: {done}")

        if done:
            print(f"Solution found in {total_steps} steps!")
            time.sleep(2)  # Pause to show final state
            break

        clock.tick(2)  # Limit to 2 FPS to see the movement

    env.close()
    return total_steps


if __name__ == "__main__":
    steps = main()
    print(f"Total steps taken: {steps}")
