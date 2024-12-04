# Multi-Agent Reinforcement Learning for Token Pruning in Vision Transformers

## Introduction

Token pruning is a crucial technique for optimizing the inference of Vision Transformers (ViTs). While ViTs are powerful, they often suffer from high computational costs, making them less practical for real-time or resource-constrained applications. By pruning redundant tokens, we can significantly reduce the number of floating-point operations (GFLOPs) without severely impacting model accuracy. This project explores a novel approach using **Multi-Agent Reinforcement Learning (MARL)** for token pruning in ViTs.

## Methodology

### Pruning Strategy

The proposed method employs two separate agents, each trained using Proximal Policy Optimization (PPO), to decide which tokens to prune based on:
- **Attention Scores**: Highlighting tokens critical to the model's prediction.
- **Similarity Scores**: Identifying redundant tokens based on token similarity.

Both agents collaborate to determine the tokens to keep, ensuring the pruning aligns with a target prune ratio. A custom reward system balances classification accuracy and computational efficiency.

### Overview of MARL Pruning Pipeline

1. **Multi-Agent System**:
    - **Attention Agent**: Focuses on preserving tokens with higher attention weights.
    - **Similarity Agent**: Identifies and removes redundant tokens using similarity metrics.
2. **Environment**:
    - Simulates the token pruning process and evaluates the agents' decisions using classification accuracy and deviation from the desired prune ratio.
3. **Training**:
    - Agents are optimized using PPO with a reward system encouraging efficient token selection.
  

## Results

The results below compare the baseline model, greedy pruning, random pruning, and MARL pruning. Metrics include GFLOPs, accuracy, and F1 score.

| Method            | GFLOPs | Accuracy (%) | F1 Score |
|--------------------|--------|--------------|----------|
| **Baseline**       | 1.17   | 81.63        | 0.81     |
| **Greedy Pruning** | 0.56   | 55.12        | 0.55     |
| **Random Pruning** | 0.56   | 68.38        | 0.68     |
| **MARL Pruning**   | 0.65   | 74.54        | 0.74     |

The MARL-based approach achieves a significant reduction in computational cost with minimal impact on accuracy and F1 score, outperforming greedy and random pruning techniques.

## Conclusion

This project demonstrates the potential of **Multi-Agent Reinforcement Learning (MARL)** in token pruning for Vision Transformers. By integrating attention and similarity-based pruning strategies, the MARL approach balances computational efficiency and model performance. Future work can explore dynamic target prune ratios and adapting the methodology to other transformer-based architectures.
