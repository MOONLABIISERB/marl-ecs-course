# Mid Semester Report

**Name:** Gavit Deepesh Ravikant  
**Roll No:** 20114  

## Method Used: Q-Learning


###  Run Code
```
python main.py

```
### Results

Below is the episode vs cumulative reward plot that illustrates the training convergence.

![Episode vs. Cumulative Reward](https://github.com/MOONLABIISERB/marl-ecs-course/blob/gavit_20114/MidSem/episode_rewards_plot.png) <!-- Ensure the link is accessible -->


| Metric                       | Value       |
|------------------------------|-------------|
| Total Episodes                | 10,000      |
| Average Reward                | 275.53       |
| Maximum Reward Achieved       | 515.05          |
| Final Epsilon (Exploration Rate) | 0.0100   |

### Conclusion

The Q-Learning agent effectively tackled the Modified Traveling Salesman Problem (ModTSP) by learning to navigate through targets efficiently over 10,000 training episodes. The agent exhibited significant performance improvements, as shown by the increasing average rewards and achieving a high maximum reward. The implementation of an epsilon-greedy policy facilitated a balanced exploration and exploitation strategy, enabling the discovery of optimal routes.

#### Key Observations:

1. **Performance Enhancement:** The consistent improvement in rewards indicates successful learning and policy optimization.
2. **Epsilon Decay Effectiveness:** The gradual reduction of the exploration rate allowed the agent to focus on exploiting learned strategies while maintaining minimal exploration.
3. **Scalability Considerations:** Although the agent performed well with 10 targets, scaling to larger problem sizes may necessitate advanced techniques such as state abstraction or deep reinforcement learning.

#### Future Enhancements:

- **State Representation Improvements:** Opportunities exist to enhance state representations for better learning.
- **Advanced Algorithms:** Incorporating more sophisticated algorithms (e.g., Deep Q-Networks) can potentially improve performance.
- **Dynamic Elements:** Introducing dynamic elements can further test and enhance the agent's capabilities.

Overall, this project validates the applicability of Q-Learning in solving combinatorial optimization problems like the Traveling Salesman Problem, highlighting its potential for developing intelligent routing and planning solutions.
