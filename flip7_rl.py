# Flip 7 Reinfocement Learning File
# Reinforcement Learning components: Q-table, State representation,
# Reward function, action selection, Q-update rule

import random
# Claude AI recommended and replaced some of the q-table dictionary logic into a binary file so I don't lose my data when I close my python instance.
# this is why we are utilizing pickle - not something I have fully understood.
import pickle
from collections import defaultdict
from typing import Tuple
from flip7_engine import PlayerState, Flip7RoundEngine


# ----------------------------
# Hyperparameters
# ----------------------------

ALPHA         = 0.7     # learning rate — aggressive updates from new experience
GAMMA         = 0.63    # discount factor — slightly future-leaning
EPSILON       = 1.0     # exploration rate — starts fully exploratory
EPSILON_DECAY = 0.9996  # decay per game — hits floor around game 8000
EPSILON_MIN   = 0.05    # floor — agent never stops exploring entirely


# ----------------------------
# Q-table
# Structure: {state_tuple: {"DRAW": float, "STOP": float}}
# ----------------------------

q_table = defaultdict(lambda: {"DRAW": 0.0, "STOP": 0.0})


# ----------------------------
# State representation
# ----------------------------

def get_score_bin(round_score: int) -> int:
    """
    Final tuned score bins.
    Decision-aware score bins concentrated around heuristic thresholds.
    Boundaries informed by empirical baseline avg banking points:
        Conservative: 21.6, Balanced: 29.7, Greedy: 34.0
    Anything over 49 collapsed into bin 7 — rarely reached, behavior obvious.
    """
    if round_score <= 12:    return 0   # just draw, no real decision
    elif round_score <= 18:  return 1   # approaching conservative threshold
    elif round_score <= 23:  return 2   # approaching balanced threshold
    elif round_score <= 29:  return 3   # approaching greedy threshold
    elif round_score <= 36:  return 4   # high risk zone
    elif round_score <= 42:  return 5   # very high
    elif round_score <= 49:  return 6   # extreme
    else:                    return 7   # 50+ almost certainly stop


def get_state(player: PlayerState, engine: Flip7RoundEngine,
              player_idx: int) -> Tuple:
    """
    Converts current game situation into a hashable state tuple for Q-table.
    Bins the state space so the state space stays manageable
    State: (score_bin, cards_drawn, gap_bin, has_second_chance)
    
    score_bin:        binned round score — 8 decision-aware bins
    cards_drawn:      number of cards drawn this round (uncapped)
    gap_bin:          positional awareness vs leading opponent
                      0=ahead, 1=even, 2=moderately behind, 3=significantly behind
    has_second_chance: 1 if second chance available, 0 otherwise

    Updated 4.9.26 — replaced raw round score with score_bin to reduce
    state space and focus on strategic thresholds rather than exact points.
    """
    score_bin         = get_score_bin(player.round_points)
    cards_drawn       = player.cards_drawn_count
    has_second_chance = int(player.second_chance_available)

    leading_idx = engine.get_leading_opponent(player_idx)
    if leading_idx is not None:
        leading_score = (engine.players[leading_idx].total_points +
                        engine.players[leading_idx].round_points)
        my_score = player.total_points + player.round_points
        raw_gap  = leading_score - my_score  # positive = behind
    else:
        raw_gap = 0

    if raw_gap > 45:      gap_bin = 3   # significantly behind
    elif raw_gap > 15:    gap_bin = 2   # moderately behind
    elif raw_gap >= -15:  gap_bin = 1   # roughly even
    else:                 gap_bin = 0   # ahead

    return (score_bin, cards_drawn, gap_bin, has_second_chance)


# ----------------------------
# Reward function
# ----------------------------

def compute_reward(points_banked: int, busted: bool,
                   score_gap: int,
                   win_threshold: int = 200) -> float:
    """
    Hybrid reward combining normalized score accumulation,
    positional awareness, and tiered bust penalties.

    score_gap: leading opponent total - agent total
               positive = agent is BEHIND
               negative = agent is AHEAD

    Tiered bust penalties only apply greed tax when agent is ahead.
    Trailing agents receive standard penalty to avoid punishing
    legitimate comeback attempts.

    Updated 4.9.26 — rescaled score_component from /200 to /50
    to balance positive rewards against bust penalty magnitudes.
    Updated 4.30.26 - added more logic to add weight to 
    """
    if busted:
        if score_gap <= -45:    return -0.6    # ahead by 45+ — greed tax
        elif score_gap <= -20:  return -0.45   # ahead by 20-44
        else:                   return -0.3    # trailing or standard bust

    score_component    = points_banked / 50
    position_component = score_gap / win_threshold
    reward = 0.6 * score_component + 0.4 * (-position_component)

    return min(reward, 1.0)


# ----------------------------
# Action selection
# ----------------------------

def select_action(state: Tuple, epsilon: float) -> str:
    """
    Epsilon-greedy action selection.
    
    Forced draw floor: agent must draw if score bin is 0 or
    cards drawn < 2 — prevents pathological early stopping.
    
    With probability epsilon: explore (random action)
    With probability 1-epsilon: exploit (best known Q-value)
    """
    score_bin, cards_drawn, gap_bin, has_second_chance = state

    # Forced draw floor — override Q-table in obvious draw territory
    if cards_drawn < 2 or score_bin == 0:
        return "DRAW"

    if random.random() < epsilon:
        return random.choice(["DRAW", "STOP"])

    q_values = q_table[state]
    if q_values["DRAW"] >= q_values["STOP"]:
        return "DRAW"
    return "STOP"


# ----------------------------
# Q-table update
# ----------------------------

def update_q_table(state: Tuple, action: str, reward: float,
                   next_state: Tuple, terminal: bool) -> None:
    """
    Applies Bellman update to Q-table.

    terminal: True when agent busted or game ended.
              Drops future reward component — update purely from experience.

    Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s')) - Q(s,a))
    """
    current_q = q_table[state][action]

    if terminal:
        target = reward
    else:
        best_next_q = max(q_table[next_state].values())
        target      = reward + GAMMA * best_next_q

    q_table[state][action] = current_q + ALPHA * (target - current_q)


# ----------------------------
# Q-table persistence
# ----------------------------
# This is Claude.ai logic to save qtable data.
# Not expected to need to read the raw files, so this should be fine.
def save_q_table(filepath: str = "q_table.pkl") -> None:
    """Saves trained Q-table to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(dict(q_table), f)
    print(f"Q-table saved: {len(q_table)} states → {filepath}")


def load_q_table(filepath: str = "q_table.pkl") -> None:
    """Loads trained Q-table from disk into global q_table."""
    global q_table
    with open(filepath, "rb") as f:
        loaded = pickle.load(f)
    q_table = defaultdict(lambda: {"DRAW": 0.0, "STOP": 0.0}, loaded)
    print(f"Q-table loaded: {len(q_table)} states from {filepath}")
