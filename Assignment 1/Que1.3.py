import matplotlib.pyplot as plt
import networkx as nx

# Define the states, actions, transition probabilities, and rewards
states = ['Hostel', 'Academic Building', 'Canteen']
actions = ['Attend Class', 'Eat']

transition_probabilities = {
    'Hostel': {
        'Attend Class': [('Academic Building', 0.5, 3), ('Hostel', 0.5, -1)],
        'Eat': [('Canteen', 1.0, 1)]
    },
    'Academic Building': {
        'Attend Class': [('Academic Building', 0.7, 3), ('Canteen', 0.3, 1)],
        'Eat': [('Canteen', 0.8, 1), ('Academic Building', 0.2, 3)]
    },
    'Canteen': {
        'Attend Class': [('Academic Building', 0.6, 3), ('Hostel', 0.3, -1), ('Canteen', 0.1, 1)],
        'Eat': [('Canteen', 1.0, 1)]
    }
}

# Create a directed graph
G = nx.DiGraph()

# Add nodes for states
for state in states:
    G.add_node(state)

# Add edges for transitions
for state in states:
    for action in actions:
        for next_state, prob, reward in transition_probabilities[state][action]:
            G.add_edge(state, next_state, label=f'{action}\nP={prob}, R={reward}')

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # positions for all nodes

plt.figure(figsize=(10, 7))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', edgecolors='black')

# Draw edges with labels
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title('MDP Transition Graph')
plt.show()
