# ENDSEM

# Mini Football

## Problem Statement

* To develop and optimize a multi-agent reinforcement learning (MARL) policy for playing mini football.

Challenges:

- Cooperative & Competitive Strategies: To make the agents cooperatively and competitively learn strategies to score goals, dribble, pass, defence, and adapting to the actions of the opponent.

- Individual and Collective Learning: Each agent should learn to control the ball individually and also coordinate with teammates to keep possession of the ball with its own team.

- Continuous State Space: Agents must learn from an infinite possibilities of positions, velocities, and orientations on the field. It is difficult to map each combination of state variables to a particular optimal action.

● We aim to create autonomous players (agents) that can learn complex behaviours such as teamwork, strategy, and optimal decision-making in dynamic, rapidly changing environments.

## Setup

### Environment

- 3v3 mini football match

- Team that scores goal wins

- Episode ends if ball out of pitch


### Agent

- Dribble: Move with the ball toward the goal.

- Shoot: Shoot the ball toward the goal.

- Pass: Shoot the ball toward a teammate.

- Move: Move without the ball to a more strategic position.

### Goal

To score a GOAL

![alt text](https://github.com/MOONLABIISERB/marl-ecs-course/blob/rugved_21294/ENDSEM/download%20(3).jpg)

### Rewards

![alt text](https://github.com/MOONLABIISERB/marl-ecs-course/blob/rugved_21294/ENDSEM/Screenshot%202024-11-27%20193922.png)

## Observations

1.  In early episodes, blue agents are not wise whereas red agents know how to play. Therefore they goal. Defending them gives positive reward to our agents. So blue agents learn to defend, whereas their attacking learning is very poor.
Thus, over some iterations, Blue agents become better defenders.

2.  In earlier experiments, it was observed that the agents tend to go towards the borders and corners of the pitch. The border penalty was added to mitigate this issue.

