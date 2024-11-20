import torch
import numpy as np
from torch.nn import functional as F
from assignments.a3.env import GridWorldEnv, Action, Agent
from assignments.a3.dqn import Q_Network
from random import choices


def update_target_network(q_net: Q_Network, target_net: Q_Network):
    target_net.load_state_dict(state_dict=q_net.state_dict())


def train(
    env: GridWorldEnv, epsilon: float = 0.5, max_iters: int = 1_000, gamma: float = 0.99
) -> Q_Network:
    # Create the networks
    q_net = Q_Network(
        n_actions=len(Action), fov_x=env._agent_fov[0], fov_y=env._agent_fov[1]
    )
    optimizer = torch.optim.Adam(params=q_net.parameters())
    loss_fn = torch.nn.HuberLoss()

    target_net = Q_Network(
        n_actions=len(Action), fov_x=env._agent_fov[0], fov_y=env._agent_fov[1]
    )
    update_target_network(q_net=q_net, target_net=target_net)
    target_net.eval()

    for t in range(max_iters):
        s = torch.FloatTensor(env.reset())
        while True:
            explore = np.random.random() < epsilon
            q = q_net(torch.FloatTensor(s))
            print(q)
            if explore:
                a = choices(population=list(Action), k=env.n_agents)
                print(a)
            else:
                action_probas = torch.softmax(q, dim=1).detach().numpy()
                action_indexes = action_probas.argmax(axis=1)
                a = [Action(value=index) for index in action_indexes]
                print(a)
            s_, r, dones, done, terminated = env.step(actions=a)

            q_hat = target_net(torch.FloatTensor(s_))
            print(q_hat)
            print(q_hat.max(dim=1))
            y = r + gamma * torch.max(q_hat, dim=1, keepdim=True).values * (
                1.0 - (done or terminated)
            )
            loss = loss_fn(y, q)
            print(f"Loss = {loss} | Reward = {r}")

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Agent coords: {env.agent_coords}, {done=}, {terminated=}")
            if done or terminated:
                break

    return q_net


def main():
    # Create environment
    walls = {
        (0, 4),
        (1, 4),
        (2, 4),
        (2, 5),
    }
    agents = [
        (1, 1),
        # (6, 2),
    ]

    goals = [
        (5, 8),
        # (6, 4),
    ]

    env = GridWorldEnv(
        width=10,
        height=10,
        agents_start=agents,
        walls=walls,
        goals=goals,
        agent_fov=(2, 2),
        max_steps=100,
    )

    # # Create the networks
    # q_net = Q_Network(
    #     n_actions=len(Action), fov_x=env._agent_fov[0], fov_y=env._agent_fov[1]
    # )
    # target_net = Q_Network(
    #     n_actions=len(Action), fov_x=env._agent_fov[0], fov_y=env._agent_fov[1]
    # )
    # target_net.load_state_dict(state_dict=q_net.state_dict())
    # target_net.eval()

    # state, reward, dones, done, terminated = env.step(actions=[Action.RIGHT])
    # print(f"state shape = {state.shape}")
    # y = q_net(torch.FloatTensor(state))
    # print(f"y = {torch.softmax(y, dim=1)}")

    train(env=env, epsilon=1.0, max_iters=50)


if __name__ == "__main__":
    main()
