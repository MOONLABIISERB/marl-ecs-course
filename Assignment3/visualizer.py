import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter   
import numpy as np
     
def gif_render(env,ep,total_steps,):
    ret = 0
    _ = env.reset()
    ep_rets = []
    rectangle1 = matplotlib.patches.Rectangle((5,0),width=1,height=2,facecolor='gray',edgecolor='gray')
    rectangle2 = matplotlib.patches.Rectangle((4,2),width=2,height=1,facecolor='gray',edgecolor='gray')
    rectangle3 = matplotlib.patches.Rectangle((0,5),width=2,height=1,facecolor='gray',edgecolor='gray')
    rectangle4 = matplotlib.patches.Rectangle((2,4),width=1,height=2,facecolor='gray',edgecolor='gray')
    rectangle5 = matplotlib.patches.Rectangle((7,4),width=1,height=2,facecolor='gray',edgecolor='gray')
    rectangle6 = matplotlib.patches.Rectangle((8,5),width=2,height=1,facecolor='gray',edgecolor='gray')
    rectangle7 = matplotlib.patches.Rectangle((4,7),width=2,height=1,facecolor='gray',edgecolor='gray')
    rectangle8 = matplotlib.patches.Rectangle((4,8),width=1,height=2,facecolor='gray',edgecolor='gray')
    rectangle9 = matplotlib.patches.Rectangle((1,1),width=1,height=1,facecolor='blue',edgecolor='blue')
    rectangle10 = matplotlib.patches.Rectangle((8,1),width=1,height=1,facecolor='yellow',edgecolor='yellow')
    rectangle11 = matplotlib.patches.Rectangle((1,8),width=1,height=1,facecolor='green',edgecolor='green')
    rectangle12 = matplotlib.patches.Rectangle((8,8),width=1,height=1,facecolor='purple',edgecolor='purple')
    fig = plt.figure()
    ax = plt.gca()
    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    ax.add_patch(rectangle3)
    ax.add_patch(rectangle4)
    ax.add_patch(rectangle5)
    ax.add_patch(rectangle6)
    ax.add_patch(rectangle7)
    ax.add_patch(rectangle8)
    ax.add_patch(rectangle9)
    ax.add_patch(rectangle10)
    ax.add_patch(rectangle11)
    ax.add_patch(rectangle12)
    l1, = plt.plot([], [], 'g--')
    l2, = plt.plot([], [], 'm--')
    l3, = plt.plot([], [], 'b--')
    l4, = plt.plot([], [], 'y--')
    p1, = plt.plot([], [], 'go')
    p2, = plt.plot([], [], 'mo')
    p3, = plt.plot([], [], 'bo')
    p4, = plt.plot([], [], 'yo')
    plt.title('MAP')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid()
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 11, 1))
    writer = PillowWriter(fps=7)
    xlist1 = []
    xlist2 = []
    xlist3 = []
    xlist4 = []
    ylist1 = []
    ylist2 = []
    ylist3 = []
    ylist4 = []
    with writer.saving(fig, outfile='gifs/'+f"visual.gif", dpi = 100):
        for steps in range(total_steps):
            Q = np.load('Q_vals.npy',allow_pickle=True).item()
            actions=[np.argmax(Q[agent][env.agent_currentloc[agent][0],env.agent_currentloc[agent][1],:]).item() for agent in env.agents]
            actions=list(np.random.randint(low=0,high=5,size=(4)))
            print('actions taken:',actions)
            state, reward, terminated, info = env.step(actions)
            xlist1.append(env.agent_currentloc['green'][0]+0.5)
            ylist1.append(9-env.agent_currentloc['green'][1]+0.5)
            xlist2.append(env.agent_currentloc['purple'][0]+0.5)
            ylist2.append(9-env.agent_currentloc['purple'][1]+0.5)
            xlist3.append(env.agent_currentloc['blue'][0]+0.5)
            ylist3.append(9-env.agent_currentloc['blue'][1]+0.5)
            xlist4.append(env.agent_currentloc['yellow'][0]+0.5)
            ylist4.append(9-env.agent_currentloc['yellow'][1]+0.5)
            l1.set_data(xlist1,ylist1)
            l2.set_data(xlist2,ylist2)
            l3.set_data(xlist3,ylist3)
            l4.set_data(xlist4,ylist4)
            p1.set_data(env.agent_currentloc['green'][0]+0.5,9-env.agent_currentloc['green'][1]+0.5)
            p2.set_data(env.agent_currentloc['purple'][0]+0.5,9-env.agent_currentloc['purple'][1]+0.5)
            p3.set_data(env.agent_currentloc['blue'][0]+0.5,9-env.agent_currentloc['blue'][1]+0.5)
            p4.set_data(env.agent_currentloc['yellow'][0]+0.5,9-env.agent_currentloc['yellow'][1]+0.5)
            writer.grab_frame()
            if np.all([env.terminations[agent]==True for agent in env.agents]):
                            done = True
            else:
                done = False 
            for agent in env.agents:
                ret += reward[agent]
            if done:
                break
    print(f"Episode Return: {ret}")
