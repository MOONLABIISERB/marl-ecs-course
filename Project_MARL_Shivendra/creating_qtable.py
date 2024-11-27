import pickle

q_table = {}

pickle_p = open("pursuer_qtable.pickle","wb")
pickle.dump(q_table, pickle_p)
pickle_p.close()

pickle_e = open("evader_qtable.pickle","wb")
pickle.dump(q_table, pickle_e)
pickle_e.close()