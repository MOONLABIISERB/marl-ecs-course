import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_valueIteration(states, values, policies):
    x_coords = []
    y_coords = []
    v = []
    p = []
    for state in states:
        parts = state.split('_')
        x_coords.append(int(parts[1]))
        y_coords.append(int(parts[2]))

        v.append(values[state])
        p.append(policies[state])
    

    grid_size_x = 9
    grid_size_y = 9

    # Initialize value and policy grid
    grid_v = np.full((grid_size_y, grid_size_x), np.nan)
    grid_p = np.full((grid_size_y, grid_size_x), "", dtype=str)
    
    # Fill values for value and policy grid based on Value Iteration with C#
    for x,y,val in zip(x_coords, y_coords, v):
        grid_v[y,x] = val

    for x,y, policy in zip(x_coords, y_coords, p):
        grid_p[y,x] = policy

    fig, axes = plt.subplots(1, 2, figsize=(36, 18))

    fig.suptitle('Value Iteration', fontsize=16)

    # Plot Values 
    axes[0].imshow(grid_v, origin='lower')
    axes[0].set_title('Values')

    for y in range(grid_size_y):
        for x in range(grid_size_x):
            if not np.isnan(grid_v[y,x]):               
                axes[0].text(x,y, f'{grid_v[y,x]:.3e}', ha='center', va='center', color='white')

    # Plot policies
    axes[1].imshow(grid_v, origin='lower')
    axes[1].set_title("Policies")

    for y in range(grid_size_y):
        for x in range(grid_size_x):             
            if not grid_p[y,x] == "":
                plt.text(x,y, grid_p[y,x], ha='center', va='center', color='white')

    plt.show()

# Same as value iteration
def plot_policyIteration(states, values, policies):
    x_coords = []
    y_coords = []
    v = []
    p = []
    for state in states:
        parts = state.split('_')
        x_coords.append(int(parts[1]))
        y_coords.append(int(parts[2]))

        v.append(values[state])
        p.append(policies[state])
    

    grid_size_x = 9
    grid_size_y = 9

    grid_v = np.full((grid_size_y, grid_size_x), np.nan)
    grid_p = np.full((grid_size_y, grid_size_x), "", dtype=str)
    
    for x,y,val in zip(x_coords, y_coords, v):
        if (val == 0):
            grid_v[y,x] = np.nan
        else:
            grid_v[y,x] = val

    for x,y, policy in zip(x_coords, y_coords, p):
        grid_p[y,x] = policy

    fig, axes = plt.subplots(1, 2, figsize=(36, 18))

    fig.suptitle('Policy Iteration', fontsize=16)

    axes[0].imshow(grid_v, origin='lower')
    axes[0].set_title('Values')

    for y in range(grid_size_y):
        for x in range(grid_size_x):
            if not np.isnan(grid_v[y,x]):               
                axes[0].text(x,y, f'{grid_v[y,x]:.3e}', ha='center', va='center', color='white')

    axes[1].imshow(grid_v, origin='lower')
    axes[1].set_title("Policies")

    for y in range(grid_size_y):
        for x in range(grid_size_x):             
            if not grid_p[y,x] == "":
                plt.text(x,y, grid_p[y,x], ha='center', va='center', color='white')

    plt.show()