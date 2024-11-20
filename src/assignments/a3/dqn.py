import torch
from torch.nn import functional as F


class Q_Network(torch.nn.Module):
    def __init__(self, n_actions: int, fov_x: int, fov_y: int) -> None:
        super().__init__()

        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=2, stride=1, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1
        )

        # Calculate flattened feature size dynamically
        # Create a dummy input to compute the output shape after convolutions
        dummy_input = torch.zeros((1, 3, 2 * fov_y + 1, 2 * fov_x + 1))
        with torch.no_grad():
            dummy_output = self.conv3(self.conv2(self.conv1(dummy_input)))
        flattened_size = dummy_output.numel()  # Total number of elements in the tensor

        # Fully connected layers
        self.fc1 = torch.nn.Linear(in_features=flattened_size, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=n_actions)

    def forward(self, x: torch.Tensor):
        # Input shape: (n_agents, 3, fov_y, fov_x)
        n_agents = x.shape[0]

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))  # (n_agents, 32, fov_y, fov_x)
        x = F.relu(self.conv2(x))  # (n_agents, 64, fov_y, fov_x)
        x = F.relu(self.conv3(x))  # (n_agents, 64, fov_y, fov_x)

        # Flatten feature maps for each agent
        x = x.view(n_agents, -1)  # (n_agents, flattened_size)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))  # (n_agents, 128)
        q_values = self.fc2(x)  # (n_agents, n_actions)

        return q_values  # Q-values for each agent's actions


def main():
    x = torch.rand(2, 3, 3, 3)  # Batch size = 2, 3 channels, 2x2 field of view
    print(f"{x=}")
    # with torch.no_grad():
    qnet = Q_Network(n_actions=5, fov_x=1, fov_y=1)  # Correct field of view
    loss_fn = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(lr=1e-3, params=qnet.parameters())
    for i in range(30):
        y = qnet(x)
        prob = torch.softmax(y, dim=1)
        print(f"{prob=}")
        y = torch.tensor(
            [
                [0.5, 0.2, 0.3, 0.0, 0.0],
                [0.4, 0.1, 0.2, 0.1, 0.2],
            ],
        )
        loss = loss_fn(prob, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    main()
