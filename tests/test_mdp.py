from typing import List
from util.rl.mdp import MarkovDecisionProcess
from util.rl.mdp.params import State


def test_student_mdp_solver():
    states: List[State] = [
        State(name=name) for name in ["hostel", "canteen", "academic_building"]
    ]
    print(states)
