import pickle
pickle_in = open("game_logs.pickle","rb")
# pickle_in = open("evader_qtable.pickle","rb")
qtable = pickle.load(pickle_in)
print(qtable)