from sokoban_environment import SokobanEnvironment
from sokoban_agent import SokobanAgent

def main():
    env = SokobanEnvironment()
    env.display_grid()

    agent_dp = SokobanAgent(env)
    agent_dp.value_iteration()
    print("Value Function after Value Iteration:")
    print("{:<15} {:<15}".format("State", "Value"))
    print("-" * 30)
    for state, value in agent_dp.value_function.items():
        print(f"{str(state):<15} {value:<15.5f}")
    
    print("\nOptimal Policy after Value Iteration:")
    print("{:<15} {:<15}".format("State", "Optimal Action"))
    print("-" * 30)
    for state, action in agent_dp.policy.items():
        print(f"{str(state):<15} {action:<15}")

    env.reset()

    agent_mc = SokobanAgent(env)
    agent_mc.monte_carlo_control(num_episodes=1000)
    print("\nValue Function after Monte Carlo Control:")
    print("{:<15} {:<15}".format("State", "Value"))
    print("-" * 30)
    for state, value in agent_mc.value_function.items():
        print(f"{str(state):<15} {value:<15.5f}")

    print("\nOptimal Policy after Monte Carlo Control:")
    print("{:<15} {:<15}".format("State", "Optimal Action"))
    print("-" * 30)
    for state, action in agent_mc.policy.items():
        print(f"{str(state):<15} {action:<15}")

if __name__ == "__main__":
    main()
