# Mid Semester 
Agamdeep Singh 20021

![Path](tsp.gif)

### Average Profit: 
|Locations | Profits | Profit |
|---------|-----------|-------|
|Training Config| Shuffled (10 ep) | 336.82 |
| Shuffled | Shuffled | 150-250 |

## Training
```
python modified_tsp.py --mode train --model save_path.pth
```
## Inference Testing
```
python modified_tsp.py --mode test --model model_path.pth
```
**Best model**: model_15k.pth

Requirements:
- Pytorch
- wandb(for logging)
- numpy etc.


## Approach
### Method used: 
Deep Q-Network (DQN)
<br>
### Hyperparameters Summary



### DQN Architecture
| Layer Type | Dimension |
|----------------|-------|
| Input Layer(state space) | 41 units |
| Hidden Layer 1 | 256 units |
| Hidden Layer 2 | 256 units |
| Hidden Layer 3 | 256 units |
| Output Dimensions(# action) | 10 |

### Agent
| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 0.0005 |
| Gamma (Discount Factor) | 0.99 |
| Optimizer | Adam |
| Loss Function | Smooth L1 Loss |
| Gradient Clipping | Max norm 1.0 |



### Environment
| Hyperparameter | Value |
|----------------|-------|
| Number of Targets | 10 |
| Max Area | 15 |
| Shuffle Time | 10 |
| Random Seed | 42 |

### Training
| Hyperparameter | Value |
|----------------|-------|
| Number of Episodes | 10^4 |
| Max Steps per Episode | 10 |
| Replay Memory Capacity | 10^4 |
| Batch Size | 10^2 |
| Min Epsilon | 0.01 |
| Max Epsilon | 0.7 |
| Epsilon Decay Rate | 0.0005 |
| Target Network Update Frequency | Every 30 episodes |

**Best results** : Early stopping at 15k episodes.

### Environment
| Hyperparameter | Value |
|----------------|-------|
| Number of Targets | 10 (default) |
| Max Area | 15 (default) |
| Shuffle Time | 10 (default) |
| Random Seed | 42 (default) |
| Max Steps | Equal to number of targets |
| Initial Profits | Range from 10 to 100 (increments of 10) |

## Results

### Best (Early stopping - 15k steps)
![Distance](plots/profit_15k.png)

![Distance](plots/distance_15k.png)

![Distance](plots/loss_15k.png)


---
### Long 
Average Distance PER STEP

![Distance](plots/dist.png)

Average loss PER STEP
![Distance](plots/loss.png)

Average reward PER STEP
![Distance](plots/reward.png)

### Discussion of Results
- **Distance** per episode went down as training proceeding, this is a good thing as shorter paths weakly correalte to profit maximisation. This is due to the fact that longer path will allow profits to decay more.

- **Loss** Loss Has started to converge`

- **Episode Profit** 
    - The reward has actually gone up to be positive, which is amazing. 
    - This is amazing because that the agent is able to visit all cities before the profit decays enough to become negative.
    - Model is effectively estimating Q values
    

