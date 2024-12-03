from ChainReaction_environment.env.environment import ChainReactionEnvironment
env = ChainReactionEnvironment(render_mode='human')
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = info
    print(f'Action: {action} by agent: {agent}')
    env.step(action)

env.close()