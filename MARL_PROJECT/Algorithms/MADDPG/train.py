import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from ChainReaction_environment.env.environment import ChainReactionEnvironment
from utils import plot_learning_curve, plot_learning_curve2

env = ChainReactionEnvironment(render_mode=None)
env.reset()

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state
figure_file1 = 'plots/num_steps'
figure_file2 = 'plots/rewards'
if __name__ == '__main__':
    #scenario = 'simple'
    
    env.reset()
    observation=env.board

    actor_dims = []
    for i in range(2):

        actor_dims.append(observation)
    critic_dims = sum(actor_dims)

    critic_dims = critic_dims.shape

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 25    #env.action_space[0].n
    maddpg_agents = MADDPG( env, chkpt_dir='checkpoints/',
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01
                           )

    memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                        n_actions, 2, batch_size=1024)

    PRINT_INTERVAL = 100
    N_GAMES = 10000
    MAX_STEPS = 50000
    total_steps = 0
    best_score = [0,0]

    score_history = {'P1':[],'P2':[]}#{'P1':0,'P2':0}
    ep_st={}
    evaluate = False
    resume = True
    if resume:
        maddpg_agents.load_checkpoint()
    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        env.reset()
        obs= env.board
        score = [0,0]
        done = False
        episode_steps = 0

        while not done:
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            #print(done)
            #print(f'Action: {actions} by agent: {env.agent_selection}')
            if episode_steps%2==0:
                if i%10 and i<3000:
                    act=np.random.randint(0,25)

                    env.step(act)
                else:
                    env.step(actions[0])
            else:
                if i%10 and i<3000:
                    act=np.random.randint(0,25)

                    env.step(act)

                else:    
                    env.step(actions[1])
            obs_=env.board
            reward=env.rewards

            done=any(env.terminations.values())
            if done:
                ep_st[i]=episode_steps


            state = obs#obs_list_to_state_vector(obs)
            state_ = obs_#obs_list_to_state_vector(obs_)

            #print(done)
            if episode_steps >= MAX_STEPS:
                done = True

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score_history['P1'].append(env.rewards['P1'])
            score_history['P2'].append(env.rewards['P2'])
            total_steps += 1
            episode_steps += 1
            

        '''
        avg_score=[sum(col) / len(col) for col in zip(score_history.values())]
        #avg_score = {'P1':np.mean(score_history[0][0][-1:]),'P2':np.mean(score_history[0][1][-1:])}

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:

            print(f"Episode {i}: Average Player Scores: {avg_score}")

        '''     
    maddpg_agents.save_checkpoint()       
    x1 = [i+1 for i in range(len(score_history['P1']))]
    x2 = [i+1 for i in range(len(score_history['P2']))]
    plot_learning_curve2(ep_st.keys(), ep_st.values(), figure_file1)
    plot_learning_curve2(x2, score_history['P2'], figure_file2)
