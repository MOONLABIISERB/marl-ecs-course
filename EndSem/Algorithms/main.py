from ChainReaction_environment.env.environment import ChainReactionEnvironment
import torch as T
import numpy as np
#from PPO.network import ActorNetwork
from MADDPG.networks import ActorNetwork



device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
var=0
env = ChainReactionEnvironment(render_mode='human')
env.reset()   
if var==0:    

#    model=ActorNetwork(0.01,1024,'P!','chkpt')
    model=ActorNetwork(0.0003,'P!')# PPO
    #MADDPG\\checkpoints\\P0_actor
    model.load_state_dict(T.load('MADDPG\\checkpoints\\P0_actor', map_location=device))
    model.eval()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            if agent=='P1':
                
                action = info
                env.step(action)
            else:
                observation=env.board
                observation=T.tensor(observation,dtype=T.float, device=device).unsqueeze(dim=0)
                action =model.forward(observation)
                action=np.argmax(action.detach().cpu().numpy())
                print('___________________________')
                print(action)
                print('___________________________')
                env.step(action)
            
        print(f'Action: {action} by agent: {agent}')

        if termination:
            break
    env.close()

else:


    model=ActorNetwork(0.03,'P!')# PPO

    model.load_state_dict(T.load('PPO\\checkpoints\\P2\\actor_torch_ppo', map_location=device))
    model.eval()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            if agent=='P1':
                
                action = info
                env.step(action)
            else:
                observation=env.board
                observation=T.tensor(observation,dtype=T.float,device=device).unsqueeze(dim=0)
                action =model.forward(observation)
                action=np.argmax(action)
                print('___________________________')
                print(action)
                print('___________________________')
                env.step(action)
        print(f'Action: {action} by agent: {agent}')

        if termination:
            break
    env.close()
