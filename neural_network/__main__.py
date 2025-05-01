from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from poker import (
    Deck,
    pcg64,
    GameResult,
    compare_hands,
    ThreadPool,
    probability_of_winning,
)


class Phase(Enum):
    PRE_FLOP = auto()
    FLOP     = auto()
    TURN     = auto()
    RIVER    = auto()
    SHOWDOWN = auto()


@dataclass
class Player:
    starting_stack: float
    stack: float = field(init=False)
    hand: Deck   = field(init=False)
    bet: float   = field(default=0.0)

    def __post_init__(self):
        self.stack = self.starting_stack

    def post_bet(self, amount: float):
        # assume amount <= stack
        self.bet   += amount
        self.stack -= amount

    def reset_for_new_round(self):
        self.bet = 0.0
        self.hand = Deck.empty_deck()


class PokerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        thread_pool: ThreadPool,
        n_players: int = 2,
        starting_stack: float = 100.0,
        max_bet: float = 10.0,
        equity_sims: int = 50_000,
        rng: Optional[pcg64] = None,
        np_rng: Optional[np.random.Generator] = None,
    ):
        super().__init__()
        self.thread_pool     = thread_pool
        self.n_players       = n_players
        self.starting_stack  = starting_stack
        self.max_bet         = max_bet
        self.equity_sims     = equity_sims
        self.rng             = rng    or pcg64(seed=0)
        self.np_rng          = np_rng or np.random.default_rng(seed=0)

        # Action: 0=fold, 1=check/call, 2=raise
        self.action_space = spaces.Discrete(3)
        # Raise a fixed fraction of your starting stack
        self.raise_amount = starting_stack * 0.1

        # Observation includes equity too
        self.observation_space = spaces.Dict({
            "hand":      spaces.MultiBinary(52),
            "community": spaces.MultiBinary(52),
            "stack":     spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "pot":       spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "equity":    spaces.Box(0.0, 1.0,   shape=(1,), dtype=np.float32),
        })

        # initialize players list
        self.players: List[Player] = [
            Player(starting_stack) for _ in range(n_players)
        ]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], dict]:
        super().reset(seed=seed)
        # reinitialize state
        self.deck      = Deck.create_full_deck()
        self.community = Deck.empty_deck()
        self.pot       = 0.0
        for p in self.players:
            p.reset_for_new_round()
        self.phase = Phase.PRE_FLOP

        # deal hole cards
        self._deal_hole()
        # record initial stack
        self.initial_stack = self.players[0].stack
        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, np.ndarray]:
        # helper: 64-bit mask → 52-length vector
        def mask_to_bits(mask: int) -> np.ndarray:
            bits = np.zeros(52, dtype=np.int8)
            for i in range(52):
                bits[i] = 1 if ((mask >> i) & 1) else 0
            return bits

        me = self.players[0]
        obs = {
            "hand":      mask_to_bits(int(me.hand.mask)),
            "community": mask_to_bits(int(self.community.mask)),
            "stack":     np.array([me.stack], dtype=np.float32),
            "pot":       np.array([self.pot],  dtype=np.float32),
        }
        # equity estimation (optional)
        eq = probability_of_winning(
            me.hand,
            self.community,
            self.equity_sims,
            self.n_players,
            self.thread_pool
        )
        obs["equity"] = np.array([eq], dtype=np.float32)
        return obs

    def _deal_hole(self):
        for p in self.players:
            p.hand = self.deck.pop_random_cards(self.rng, 2)

    def _deal_flop(self):
        self.community = self.deck.pop_random_cards(self.rng, 3)

    def _deal_turn(self):
        c = self.deck.pop_random_card(self.rng)
        self.community.add_card(c)

    def _deal_river(self):
        c = self.deck.pop_random_card(self.rng)
        self.community.add_card(c)

    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, dict]:
        me = self.players[0]

        # 0 = fold
        if action == 0:
            reward = me.stack - self.initial_stack
            return self._get_obs(), reward, True, False, {}

        # 1 = check/call → no additional bet
        if action == 1:
            agent_bet = 0.0
        # 2 = raise
        else:
            agent_bet = min(self.raise_amount, me.stack)

        # post agent's bet
        me.post_bet(agent_bet)
        self.pot += agent_bet

        # opponent random bet
        opp = self.players[1]
        opp_stack = opp.stack
        max_opp = min(self.max_bet, opp_stack)
        opp_bet = float(self.np_rng.integers(0, int(max_opp) + 1))
        opp.post_bet(opp_bet)
        self.pot += opp_bet

        # deal community cards
        if self.phase == Phase.PRE_FLOP:
            self.phase = Phase.FLOP
            self._deal_flop()
        elif self.phase == Phase.FLOP:
            self.phase = Phase.TURN
            self._deal_turn()
        elif self.phase == Phase.TURN:
            self.phase = Phase.RIVER
            self._deal_river()
        else:
            self.phase = Phase.SHOWDOWN

        done = (self.phase == Phase.SHOWDOWN)
        info = {}

        # showdown reward
        if done:
            result = compare_hands(
                me.hand,
                self.community,
                [self.players[1].hand]
            )
            # net reward = current_stack - initial_stack
            if result == GameResult.PlayerWins:
                reward = me.stack - self.initial_stack
            elif result == GameResult.Lose:
                reward = self.initial_stack - me.stack
            else:
                reward = 0.0
            # reset pot
            self.pot = 0.0
        else:
            reward = 0.0

        return self._get_obs(), reward, done, False, info


def main():
    pool = ThreadPool(8)
    env = PokerEnv(
        thread_pool=pool,
        n_players=2,
        starting_stack=100.0,
        max_bet=10.0,
        equity_sims=50_000,
    )
    check_env(env)

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        device="cuda",
        verbose=1,
    )
    model.learn(total_timesteps=100_000)
    model.save("poker_ppo_fold_check_raise")

    # evaluation
    obs, _ = env.reset()
    for ep in range(5):
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, rew, done, _, info = env.step(action)
            total += rew
        print(f"Episode {ep}: net reward={total:.1f}, final stack={env.players[0].stack:.1f}")

if __name__ == "__main__":
    main()
