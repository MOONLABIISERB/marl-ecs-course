from marl.utils.rl.base.policy import Policy

# from tests.test_dict import Policy
from marl.utils.rl.markov.process import MarkovDecisionProcess
from marl.utils.rl.markov.solvers import MarkovDecisionProcessSolver as Solver
from typing_extensions import override
import typing as t
import numpy as np
from marl.utils.rl.base.params import State, Action


T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


class PolicyIterationSolver[T_State: State, T_Action: Action](
    Solver[T_State, T_Action]
):
    _theta: float

    def __init__(self, theta: float = 1e-6) -> None:
        super().__init__(name="ValueIterationSolver")
        self._theta = theta

    @override
    def solve(
        self, mdp: MarkovDecisionProcess[T_State, T_Action]
    ) -> t.Tuple[t.OrderedDict[T_State, float], Policy[T_State, T_Action]]:
        """
        Solve the MDP using policy iteration.
        """
        # Initialize the policy and values
        policy = Policy[T_State, T_Action](states=mdp.states, actions=mdp.actions)
        policy.determine()  # Make the policy deterministic
        v: t.OrderedDict[T_State, float] = t.OrderedDict(
            {state: np.random.randn() for state in mdp.states}
        )

        policy_stable = False
        while not policy_stable:
            # Policy evaluation
            delta: float = np.inf
            while not delta < self._theta:
                v_prev: np.ndarray = np.array(list(v.values()))
                for s0 in mdp.states:
                    # Get the action for this state according to policy
                    a: T_Action = policy[s0]
                    # Find the return value of the action according to policy
                    v[s0] = np.sum(
                        [
                            mdp.transition_proba.get_proba(s0=s0, a=a, s=s)
                            * (mdp.rewards.get(s0=s0, a=a, s=s) + mdp.gamma * v[s])
                            for s in mdp.states
                        ]
                    )
                v_curr: np.ndarray = np.array(list(v.values()))
                delta = np.max(np.abs(v_curr - v_prev))

            # Policy improvement
            policy_ = Policy[T_State, T_Action](states=mdp.states, actions=mdp.actions)
            for s0 in mdp.states:
                policy_[s0] = mdp.actions[
                    np.argmax(
                        [
                            np.sum(
                                [
                                    mdp.transition_proba.get_proba(s0=s0, a=a, s=s)
                                    * (
                                        mdp.rewards.get(s0=s0, a=a, s=s)
                                        + mdp.gamma * v[s]
                                    )
                                    for s in mdp.states
                                ]
                            )
                            for a in mdp.actions
                        ]
                    )
                ]

            # Check policy stability
            policy_stable = policy == policy_
            if not policy_stable:
                policy = policy_

        return v, policy
