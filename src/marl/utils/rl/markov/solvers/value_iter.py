from marl.utils.rl.base.policy import Policy
from marl.utils.rl.markov.process import MarkovDecisionProcess
from marl.utils.rl.markov.solvers import MarkovDecisionProcessSolver as Solver
from typing_extensions import override
import typing as t
import numpy as np
from marl.utils.rl.base.params import State, Action


T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


class ValueIterationSolver[T_State: State, T_Action: Action](Solver[T_State, T_Action]):
    _theta: float

    def __init__(self, theta: float = 1e-6) -> None:
        super().__init__(name="ValueIterationSolver")
        self._theta = theta

    @override
    def solve(
        self, mdp: MarkovDecisionProcess[T_State, T_Action]
    ) -> t.Tuple[t.OrderedDict[T_State, float], Policy[T_State, T_Action]]:
        """
        Solves the Markov Decision Process given by the tuple:
        (`states`, `actions`, `transition_proba`, `rewards`, `gamma`)

        Value iteration involves the following steps:
        1. Optimize value function
            - Set random values for each state.
            - For each state `s0`:
                - Set the value of `s0` (`v(s0)`) to the **maximum expected
                    return** from `s0` over all actions.
            - Check convergence: See if the maximum change in value of any
                state is less than a given threshold `theta`
                (a small number); Stop if `True`.
        2. Find optimal policy (deterministic)
            - For each state `s`:
                - Calculate expected returns on performing each action `a`.
                - Choose the action which maximizes the expected returns,
                    as the policy for state `s`.
        3. Return the optimal value function and the optimal policy.


            gamma (float): The discount factor.
            theta (float): The threshold to stop updating the value function.

        Returns:
            (Tuple[ValueFunc, Policy]): The optimal value function and the optimal policy.
        """
        # Randomize the value function
        v = t.OrderedDict[T_State, float](
            {state: np.random.randn() for state in mdp.states}
        )

        # Optimize value function
        delta: float = np.inf
        while not delta < self._theta:
            # Store the values of states
            v_prev = np.array(list(v.values()))
            for s0 in mdp.states:
                # Update the values
                v[s0] = np.max(
                    [
                        np.sum(
                            [
                                mdp.transition_proba.get_proba(s0=s0, a=a, s=s)
                                * (mdp.rewards.get(s0=s0, a=a, s=s) + mdp.gamma * v[s])
                                for s in mdp.states
                            ]
                        )
                        for a in mdp.actions
                    ]
                )
            v_curr = np.array(list(v.values()))
            delta = np.max(np.abs(v_curr - v_prev))

        # Get optimal policy from values
        policy = Policy[T_State, T_Action](states=mdp.states, actions=mdp.actions)
        for s0 in mdp.states:
            policy[s0] = mdp.actions[
                # Maximize over all actions
                np.argmax(
                    [
                        # Sum over all final states
                        np.sum(
                            [
                                # p(s | s0, a) * (r(s0, a, s) + gamma * v[s])
                                mdp.transition_proba.get_proba(s0=s0, a=a, s=s)
                                * (mdp.rewards.get(s0=s0, a=a, s=s) + mdp.gamma * v[s])
                                for s in mdp.states
                            ]
                        )
                        for a in mdp.actions
                    ]
                )
            ]
        return v, policy
