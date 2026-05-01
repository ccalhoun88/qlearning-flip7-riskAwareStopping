# Heuristic baseline policies and agent wrappers

from flip7_engine import PlayerState, Flip7RoundEngine


# ----------------------------
# Heuristic policies
# ----------------------------

def conservative_policy(player: PlayerState,
                        stop_threshold: int = 18,
                        max_cards: int = 4) -> str:
    """
    Stops at a low score threshold or low card count.
    Prioritizes safety over accumulation.
    Empirical avg banking: 21.6 points per stop during 3 agent baseline testing.
    """
    if player.round_points >= stop_threshold or \
       player.cards_drawn_count >= max_cards:
        return "STOP"
    return "DRAW"


def greedy_policy(player: PlayerState,
                  stop_threshold: int = 30,
                  max_cards: int = 6) -> str:
    """
    Stops only at high score threshold or high card count.
    Prioritizes accumulation over safety.
    Empirical avg banking: 34.0 points per stop.
    """
    if player.round_points >= stop_threshold or \
       player.cards_drawn_count >= max_cards:
        return "STOP"
    return "DRAW"


def balanced_policy(player: PlayerState,
                    stop_threshold: int = 26,
                    max_cards: int = 6) -> str:
    """
    Middle ground between conservative and greedy.
    Thresholds tuned empirically via baseline testing.
    Empirical avg banking: 29.7 points per stop.
    """
    if player.round_points >= stop_threshold or \
       player.cards_drawn_count >= max_cards:
        return "STOP"
    return "DRAW"


# ----------------------------
# Agent wrappers
# Adapts policies to (player, engine, idx) signature
# required by run_round and train_rl_agent
# ----------------------------

def conservative_agent(player: PlayerState,
                       engine: Flip7RoundEngine,
                       idx: int) -> str:
    return conservative_policy(player)


def greedy_agent(player: PlayerState,
                 engine: Flip7RoundEngine,
                 idx: int) -> str:
    return greedy_policy(player)


def balanced_agent(player: PlayerState,
                   engine: Flip7RoundEngine,
                   idx: int) -> str:
    return balanced_policy(player)


# ----------------------------
# Opponent configurations
# Maps num_opponents to player name lists
# ----------------------------

OPPONENT_CONFIGS = {
    2: ["RL_Agent", "Conservative", "Greedy"],
    3: ["RL_Agent", "Conservative", "Greedy", "Balanced"]
}
