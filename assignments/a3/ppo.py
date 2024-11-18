import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection for training
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Environment
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-Agent Network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Objectives
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
from matplotlib import pyplot as plt
from rich.console import Console
from rich.progress import Progress


def main():
    # Set the seed
    torch.manual_seed(0)

    # Define Hyperparameters
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    vmas_device = device

    # Sampling config
    frames_per_batch = 1_000
    n_iters = 10
    total_frams = frames_per_batch * n_iters

    # Training config
    n_epochs = 30
    minibatch_size = 400
    lr = 1e-4  # Learning rate
    max_grad_norm = 1.0  # Max norm for gradients

    # PPO config
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.9
    entropy_eps = 1e-4

    


if __name__ == "__main__":
    main()
