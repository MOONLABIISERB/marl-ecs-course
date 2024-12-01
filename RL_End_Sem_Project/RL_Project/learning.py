from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LeakyReLU, Conv2D, MaxPooling2D

from rl.agents import DQNAgent
from rl.memory import SequentialMemory  # For experience replay!


def build_model1(states, actions, h_nodes, h_act):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    for n, a in zip(h_nodes, h_act):
        model.add(Dense(n, activation=a))
    model.add(Dense(actions, activation='linear'))
    return model

def build_model(states, actions, h_nodes, h_act, use_cnn=False, conv_layers=None):
    model = Sequential()

    if use_cnn and conv_layers:
        # Add convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(conv_layers):
            if i == 0:
                # First conv layer needs the input shape
                model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=states))
            else:
                model.add(Conv2D(filters, kernel_size, activation='relu'))
            model.add(MaxPooling2D(pool_size=pool_size))
            model.add(BatchNormalization())

        model.add(Flatten())  # Flatten the feature maps
    else:
        # Use Dense layers for non-CNN input
        model.add(Flatten(input_shape=(1, states)))

    # Add dense layers
    for n, a in zip(h_nodes, h_act):
        model.add(Dense(n))
        if isinstance(a, str):  # Standard activation
            model.add(LeakyReLU(alpha=0.01) if a == 'leaky_relu' else Dense(n, activation=a))
        else:  # Custom activation
            model.add(a)
        model.add(BatchNormalization())  # Optional: normalize inputs to this layer
        model.add(Dropout(0.2))  # Optional: dropout for regularization

    # Output layer
    model.add(Dense(actions, activation='linear'))  # Linear activation for Q-values

    return model

def build_agent(model, actions, tmu, policy, ml):
    memory = SequentialMemory(limit=ml, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=100,
                   target_model_update=tmu)
    return dqn


def get_agent_path(name):
    return "agents/{}/{}.h5f".format(name, name)


def get_training_path(name):
    return "agents/{}/{}_training.json".format(name, name)


def get_test_path(name, nb_episodes):
    return 'agents/{}/{}_test_{}episodes.txt'.format(name, name, nb_episodes)
