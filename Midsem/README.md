# TSP Q-Learning Solution - Mid-Sem Exam
The solution consists of:
- An environment (`env.py`) that simulates the TSP with randomized target locations.
- A Q-learning agent (`sol.py`) that learns the optimal path to visit all waypoints while maximizing profits.

The solution is illustrated through:
- Cumulative reward progression over episodes (`Figure 1`).
- The final path traversed by the agent (`Figure 2`).

---

<<<<<<< HEAD
## Installation & Setup
=======
1. **sol.py**: This file contains the Q-learning agent's implementation and the environment interaction logic.
2. **env.py**: This file contains the TSP environment.
3. **Figure_1.png**: Visualization of the cumulative rewards over 1000 episodes.
4. **Figure_2.png**: Visualization of the agent's path over the waypoints at the final episode.
5. **README.md**: This file provides detailed instructions for running the code and understanding the results.
>>>>>>> ae2fb4f99337219e3360612fc9689e83b90842fa

### Requirements

To run the code, ensure you have the following dependencies installed:
- `gymnasium`
- `numpy`
- `matplotlib`
- `pandas`

You can install them using `pip`:
```bash
pip install gymnasium numpy matplotlib pandas
```

### Running the Code

<<<<<<< HEAD
1. Clone the repository:
   ```bash
   git clone https://github.com/MOONLABIISERB/marl-ecs-course.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Midsem
   ```
3. Run the simulation:
   ```bash
   python sol.py
   ```

Upon running the code, you will see:
- A plot of cumulative rewards earned by the agent after each episode.
- An animation of the agent's final path across the waypoints.
=======
1. Ensure the environment class `ModTSP` is correctly implemented or imported from the `env.py` file.
2. Run the `sol.py` script using the following command:

```bash
python sol.py
```
>>>>>>> ae2fb4f99337219e3360612fc9689e83b90842fa

---

## Files and Directories

### 1. `env.py`
This file defines the custom environment `ModTSP` based on Gym, where:
- `__init__()` initializes the environment with target locations, profits, and distance matrix.
- `reset()` resets the environment for a new episode.
- `step()` allows the agent to take actions and updates the environment state.

### 2. `sol.py`
This file contains the implementation of the Q-learning agent `AgentQLearner`. Key functions include:
- `choose_action()` for selecting actions using epsilon-greedy policy.
- `update_q_table()` for updating the Q-values based on the agent's experiences.
- `render_path()` for visualizing the final path taken by the agent.

### 3. Figures
- **Figure 1**: A graph of the cumulative rewards earned by the agent over 1000 episodes.
- **Figure 2**: A plot showing the final path taken by the agent across all waypoints.

### 4. Output Dictionary (Episodic Rewards)
The following dictionary captures the cumulative rewards at intervals during the training process:

```python
{
    'Rewards after episode 0': 44.52,
    'Rewards after episode 100': 191.48,
    'Rewards after episode 200': 186.58,
    'Rewards after episode 300': 199.22,
    'Rewards after episode 400': 186.64,
    'Rewards after episode 500': 135.70,
    'Rewards after episode 600': 204.31,
    'Rewards after episode 700': 186.58,
    'Rewards after episode 800': 199.22,
    'Rewards after episode 900': 204.31
}
```

---

## Results and Inference

### **Cumulative Reward Progression (Figure 1)**:
- This plot shows how the agent improves over time, as its cumulative rewards stabilize around 200 after approximately 100 episodes. The dips represent exploration steps where the agent attempts non-optimal actions but eventually converges to a stable solution.

### **Final Path Traversed (Figure 2)**:
- This figure illustrates the agent's final traversal of waypoints, demonstrating how it efficiently plans its route after learning the optimal path through exploration and exploitation.

---

## Conclusion

The Q-learning algorithm successfully navigates the TSP environment, learning an efficient path with minimal exploration as episodes progress. The solution showcases the balance between exploration (learning new routes) and exploitation (using known routes to maximize rewards).