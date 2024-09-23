import numpy as np
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from typing_extensions import override

from marl.utils.rl.base.params import Action, State
from marl.utils.rl.base.reward import RewardFuncManager
from marl.utils.rl.base.transition import TransitionProbaManager
from marl.utils.rl.markov.process import MarkovDecisionProcess
from marl.utils.rl.markov.solvers.policy_iter import PolicyIterationSolver
from marl.utils.rl.markov.solvers.value_iter import ValueIterationSolver


class StudentState(State):
    _location: str

    def __init__(self, location: str) -> None:
        super().__init__("STUDENT_STATE")
        self._location = location

    @override
    def _repr(self) -> str:
        return self._location


class StudentAction(Action):
    _act_name: str

    def __init__(self, act_name: str) -> None:
        super().__init__(name="STUDENT_ACTION")
        self._act_name = act_name

    @override
    def _repr(self) -> str:
        return self._act_name


def main():
    hostel = StudentState(location="hostel")
    canteen = StudentState(location="canteen")
    acad_bldg = StudentState(location="acad_bldg")

    states = [hostel, canteen, acad_bldg]

    eat = StudentAction(act_name="eat")
    study = StudentAction(act_name="study")

    actions = [eat, study]

    rewards = RewardFuncManager[StudentState, StudentAction](
        states=states, actions=actions
    )
    rewards.set(s0=hostel, reward=-1)
    rewards.set(s0=canteen, reward=1)
    rewards.set(s0=acad_bldg, reward=3)

    transition_proba = TransitionProbaManager[StudentState, StudentAction](
        states=states, actions=actions
    )
    transition_proba.set_proba(s0=hostel, a=eat, s=hostel, proba=1.0)
    transition_proba.set_proba(s0=hostel, a=eat, s=canteen, proba=0.0)
    transition_proba.set_proba(s0=hostel, a=eat, s=acad_bldg, proba=0.0)
    transition_proba.set_proba(s0=canteen, a=eat, s=hostel, proba=0.0)
    transition_proba.set_proba(s0=canteen, a=eat, s=canteen, proba=1.0)
    transition_proba.set_proba(s0=canteen, a=eat, s=acad_bldg, proba=0.0)
    transition_proba.set_proba(s0=acad_bldg, a=eat, s=hostel, proba=0.0)
    transition_proba.set_proba(s0=acad_bldg, a=eat, s=canteen, proba=0.8)
    transition_proba.set_proba(s0=acad_bldg, a=eat, s=acad_bldg, proba=0.2)

    transition_proba.set_proba(s0=hostel, a=study, s=hostel, proba=0.5)
    transition_proba.set_proba(s0=hostel, a=study, s=canteen, proba=0.0)
    transition_proba.set_proba(s0=hostel, a=study, s=acad_bldg, proba=0.5)
    transition_proba.set_proba(s0=canteen, a=study, s=hostel, proba=0.3)
    transition_proba.set_proba(s0=canteen, a=study, s=canteen, proba=0.1)
    transition_proba.set_proba(s0=canteen, a=study, s=acad_bldg, proba=0.6)
    transition_proba.set_proba(s0=acad_bldg, a=study, s=hostel, proba=0.0)
    transition_proba.set_proba(s0=acad_bldg, a=study, s=canteen, proba=0.3)
    transition_proba.set_proba(s0=acad_bldg, a=study, s=acad_bldg, proba=0.7)

    mdp = MarkovDecisionProcess[StudentState, StudentAction](
        states=states,
        actions=actions,
        transition_proba=transition_proba,
        rewards=rewards,
        gamma=0.9,
    )

    # Create the solvers
    solver_value_iter = ValueIterationSolver[StudentState, StudentAction]()
    solver_policy_iter = PolicyIterationSolver[StudentState, StudentAction]()

    # Solve the MDP
    v_value_iter, policy_value_iter = solver_value_iter.solve(mdp=mdp)
    v_policy_iter, policy_policy_iter = solver_policy_iter.solve(mdp=mdp)

    # Print out results
    console = Console()
    content_value_iter = Pretty(
        {
            "values": {
                state._location: float(np.round(value, 3))
                for state, value in v_value_iter.items()
            },
            "policy": {
                state._location: policy_value_iter[state]._act_name for state in states
            },
        }
    )
    content_policy_iter = Pretty(
        {
            "values": {
                state._location: float(np.round(value, 3))
                for state, value in v_policy_iter.items()
            },
            "policy": {
                state._location: policy_policy_iter[state]._act_name for state in states
            },
        }
    )
    columns = Columns(
        [
            Panel(content_value_iter, title="Value Iteration", expand=True),
            Panel(content_policy_iter, title="Policy Iteration", expand=True),
        ],
        equal=True,
        align="center",
        title="Results",
    )

    console.print(columns)


if __name__ == "__main__":
    main()
