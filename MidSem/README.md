# Mid Semester Report

**Name:** Gavit Deepesh Ravikant  
**Roll No:** 20114  

## Method Used: Q-Learning

### Requirements
~~~
pip install requirements.txt
~~~

###  Run Code
```
python q_learning.py

```
### Results

- 1) Below is the episode vs cumulative reward plot that illustrates the training convergence.

![Episode vs. Cumulative Reward](https://github.com/MOONLABIISERB/marl-ecs-course/blob/gavit_20114/MidSem/episode_rewards_plot.png) <!-- Ensure the link is accessible -->


| Metric                       | Value       |
|------------------------------|-------------|
| Total Episodes                | 10,000      |
| Average Reward                | 275.53       |
| Maximum Reward Achieved       | 515.05          |
| Final Epsilon (Exploration Rate) | 0.0100   |


- 2) Lets train and evaluate on different hyperparameters.



### Models Performance on various Hypreparameters

- Run Code
```
python train_evaluate.py

```

The following hyperparameters were tested to evaluate performance:

| Learning Rate | Discount Factor | Epsilon Decay |
|---------------|-----------------|----------------|
| 0.01          | 0.85            | 0.9995         |
| 0.01          | 0.9             | 0.999          |
| 0.05          | 0.85            | 0.999          |
| 0.1           | 0.9             | 0.995          |


 convergence graph of all models test on vairous hyperparameters

 ![Episode vs. Cumulative Reward](https://github.com/MOONLABIISERB/marl-ecs-course/blob/gavit_20114/MidSem/all_models_training_rewards.png)

 test reward comparision of all models

 ![Episode vs. Cumulative Reward](https://github.com/MOONLABIISERB/marl-ecs-course/blob/gavit_20114/MidSem/test_reward_comparison.png)

# Test Best Model
```
python test.py

```

### Conclusion

The Q-Learning agent effectively solved the Modified Traveling Salesman Problem (ModTSP) by training over 10,000 episodes with various hyperparameter settings. Model 3 (Learning Rate: 0.05, Discount Factor: 0.85, Epsilon Decay: 0.999) outperformed others, achieving the highest average test reward of 515.05 with smoother convergence.



#### Key Insights:
**Hyperparameter Impact:** Model 3 balanced learning speed and stability, showing that careful tuning of learning rates and discount factors is crucial. Higher learning rates, as seen in Model 4, led to instability.

**Epsilon Decay:** A gradual epsilon decay allowed for early exploration and late-stage exploitation, ensuring that Models 1 and 3 maintained steady performance.
Convergence:

**Convergence:** Model 3 converged faster and more smoothly, whereas Model 4 struggled due to high learning rates.

**Test Reward Comparison:** Model 3 emerged as the best performer, while Models 1 and 2 were competitive but less consistent.

**Scalability:** The agent performed well on 10 targets, but larger TSP instances may require advanced methods like Deep Q-Networks (DQN) for scalability.

This highlights the importance of hyperparameter tuning and exploration-exploitation balancing for effective policy learning.
# Thank You !! 😊
