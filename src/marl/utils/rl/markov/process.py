from dataclasses import dataclass
import typing as t

from marl.utils.rl.base.params import Action, State
from marl.utils.rl.base.reward import RewardFuncManager
from marl.utils.rl.base.transition import TransitionProbaManager
from marl.utils.rl.base.policy import Policy
import numpy as np


T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


@dataclass
class MarkovDecisionProcess[T_State: State, T_Action: Action]:
    states: t.List[T_State]
    actions: t.List[T_Action]
    transition_proba: TransitionProbaManager[T_State, T_Action]
    rewards: RewardFuncManager[T_State, T_Action]
    gamma: float

    def step(self, s0: T_State, a: T_Action) -> t.Tuple[T_State, float]:
        """
        Interacts with the environment described by the MDP by performing
        `action` at `state`.

        Args:
            s0 (T_State): The state.
            a (T_Action): The action to perform at `s0`.

        Returns:
             t.Tuple[T_State, float]: The tuple containing the next state
                and the reward obtained `r(s0, a, s)`.
        """
        s = self.states[
            np.argmax(
                [self.transition_proba.get_proba(s0=s0, a=a, s=s) for s in self.states]
            )
        ]
        r = self.rewards.get(s0=s0, a=a, s=s)
        return s, r

    def generate_episode(
        self, policy: Policy[T_State, T_Action]
    ) -> t.List[t.Tuple[T_State, T_Action, float]]:
        """
        Generates an episode as a list containing (s, a, r) tuples.
        """
        episode: t.List[t.Tuple[T_State, T_Action, float]] = []
        state: T_State = np.random.choice(np.array(self.states))
        while not state.is_terminal:
            action = policy[state]
            next_state, reward = self.step(s0=state, a=action)
            episode.append((next_state, action, reward))
            state = next_state
        return episode
