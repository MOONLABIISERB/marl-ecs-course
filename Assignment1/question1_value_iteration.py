states = ('canteen','acad_building','hostel')
actions = ('study','eat')
probs = {('canteen','1','canteen','study'):0.1,('hostel','-1','canteen','study'):0.3,('acad_building','3','canteen','study'):0.6,
         ('canteen','1','canteen','eat'):1,('hostel','-1','hostel','study'):0.5,('acad_building','3','hostel','study'):0.5,
         ('canteen','1','hostel','eat'):1,('acad_building','3','acad_building','study'):0.7,('canteen','1','acad_building','study'):0.3,
         ('acad_building','3','acad_building','eat'):0.2,('canteen','1','acad_building','eat'):0.8}
rewards = {('canteen','study','canteen'):1,('canteen','study','hostel'):-1,('canteen','study','acad_building'):3,
           ('canteen','eat','canteen'):1,('hostel','study','hostel'):-1,('hostel','study','acad_building'):3,
           ('hostel','eat','canteen'):1,('acad_building','study','acad_building'):3,('acad_building','study','canteen'):1,
           ('acad_building','eat','acad_building'):3,('acad_building','eat','canteen'):1}
values = {'canteen':-1,'acad_building':-1,'hostel':-1}
policy = {'canteen':0,'acad_building':0,'hostel':0}

def max_value_action(initial_state,gamma = 0.9):
    action_value = {'study':0,'eat':0}
    for action in actions:
        sum = 0
        for final_state in states:
            if (initial_state,action,final_state) in rewards.keys():
                sum += probs[(final_state,str(rewards[(initial_state,action,final_state)]),initial_state,action)] * (rewards[(initial_state,action,final_state)] + (gamma*values[final_state]))
                action_value[action] = sum
    return action_value


theta = 0.1
delta = 1
v=0
while theta<delta:
    delta = 0
    for state in states:
        v = values[state]
        values[state] = max(max_value_action(state).values())
        delta = max(delta,abs(v-values[state]))

for state in states:
    val = max_value_action(state)
    temp = max(val.values())
    best_action = [key for key in val if val[key] == temp]

    policy[state] = best_action

print("final set of values after value iteration:",values)
print("optimal policy after value iteration:",policy)
 

