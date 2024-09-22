
# Question 1

- Implemented value iteration ('question1_value_iteration.py') and Monte Carlo with Exploring starts(first visit and every visit strategy) for Travelling Salesman Problem('question2_MC.py').

- Obtained an imperfect policy, some states were visited more than once since the agent is only discouraged with lower reward for repeating same state and not restricted.
- Value iteration converged faster for TSP with 50 states opposed to MC with exploring starts which took significantly longer to converge.  
- Value iteration is more space and time efficient in TSP(50 states) with only a single in place state value array update as opposed to MC with large number of state action pair generation and updates.
- MC with ES(exploring starts) also has an additional burden of generating episodes and large enough number of episodes have to be generated to ensure all possible state action pairs have been found for good exploration. 2500 state-action pairs are possible for TSP(50 states) increasing with O(n^2) thus requiring even greater number of episode generation for larger state spaces.
# Question 2
 
- Implemented value iteration ('question2_value_iteration.py') and Monte Carlo with Exploring starts(first visit and every visit strategy) for a robot in a grid world ('question2_MC.py').

- Grid world environment designed with 'gymnasium' library; defined constraints and rewards(0 for successfully moving box to target location and -1 at all other steps)
- Failed to obtained meaningful policy in both value iteration and MC, possible reasons being:

  1) insufficient sampling in MC,

  2) bad reward-constraint design  (the agent would push the box into a    corner to quickly terminate episode and stop incurring negative rewards),
  3) the problem might require mutiple actions per state given the agent is moving alone or pushing the box but both algorithms can only try to converge to a single policy restricting states to only have a single associated action with it.
  4) issue with code ,either in environment design or execution of the algorithms