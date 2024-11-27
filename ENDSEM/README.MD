# ENDSEM

# Mini Football

## Problem Statement

● To develop and optimize a multi-agent reinforcement learning (MARL) policy for playing mini football.

● Challenges:

- Cooperative & Competitive Strategies: To make the agents cooperatively and competitively learn strategies to score goals, dribble, pass, defence, and adapting to the actions of the opponent.

- Individual and Collective Learning: Each agent should learn to control the ball individually and also coordinate with teammates to keep possession of the ball with its own team.

- Continuous State Space: Agents must learn from an infinite possibilities of positions, velocities, and orientations on the field. It is difficult to map each combination of state variables to a particular optimal action.

● We aim to create autonomous players (agents) that can learn complex behaviours such as teamwork, strategy, and optimal decision-making in dynamic, rapidly changing environments.

## Setup

### Environment

● 3v3 mini football match
● Team that scores goal wins
● Episode ends if ball out of pitch

### Agent

Dribble: Move with the ball toward the goal.

Shoot: Shoot the ball toward the goal.

Pass: Shoot the ball toward a teammate.

Move: Move without the ball to a more strategic position.

### Goal

To score a GOAL

![alt text](https://www.google.com/imgres?q=football%20goal%20ronaldo&imgurl=https%3A%2F%2Fimg.olympics.com%2Fimages%2Fimage%2Fprivate%2Ft_s_16_9_g_auto%2Ft_s_w960%2Ff_auto%2Fprimary%2Farwyln3kdko3mialnh8a&imgrefurl=https%3A%2F%2Folympics.com%2Fen%2Fnews%2Fcristiano-ronaldo-milestone-moments-900-career-goal&docid=7-83EIGLD2xXGM&tbnid=glgvxU9AuZVaSM&vet=12ahUKEwjUx_yI1fyJAxVusFYBHfv6L_MQM3oECBgQAA..i&w=960&h=540&hcb=2&ved=2ahUKEwjUx_yI1fyJAxVusFYBHfv6L_MQM3oECBgQAA)
