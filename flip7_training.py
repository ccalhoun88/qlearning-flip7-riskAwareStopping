# flip7_training.py
# Training loop - Q-Learning agent training against our heuristic agents

import logging
from flip7_engine import Flip7RoundEngine
from flip7_policies import (conservative_policy, greedy_policy,
                             balanced_policy, OPPONENT_CONFIGS)
from flip7_rl import (q_table, get_state, compute_reward,
                      select_action, update_q_table,
                      EPSILON_DECAY, EPSILON_MIN, save_q_table)
import flip7_rl as rl

# Training logger — metrics and snapshots
training_logger = logging.getLogger("flip7.training")

def log_training(msg: str) -> None:
    training_logger.info(msg)


# Training loop
# ----------------------------

def train_rl_agent(num_games: int = 10000,
                   num_opponents: int = 3,
                   win_threshold: int = 200,
                   verbose_every: int = 1000) -> dict:
    """
    Trains the RL agent via Q-learning against heuristic opponents.
    
    Agent plays as RL_Agent at index 0.
    Opponents follow fixed heuristic policies.
    Epsilon decays from 1.0 to EPSILON_MIN over training.
    
    Returns training statistics dict for evaluation and plotting.
    """
    # Use module-level epsilon so it persists across calls
    rl.EPSILON = 1.0

    stats = {
        "wins":           0,
        "losses":         0,
        "busts":          0,
        "win_rate_log":   [],
        "epsilon_log":    [],
        "avg_reward_log": [],
        "states_log":     []
    }

    recent_rewards = []

    for game_num in range(1, num_games + 1):

        # notice, the opponent config has been moved the heuristics pyfile
        # --- Game setup ---
        player_names = OPPONENT_CONFIGS.get(num_opponents,
                                            OPPONENT_CONFIGS[3])
        engine       = Flip7RoundEngine(player_names)
        rl_idx       = 0
        game_over    = False
        game_reward  = 0.0
        last_state   = None
        last_action  = None

        while not game_over:

            # --- Round setup ---
            engine.reset_round()
            round_over = False

            while not round_over:
                # only handles the RL agent here.
                all_done = True
                
                for idx, player in enumerate(engine.players):
                    if not player.is_active():
                        continue

                    all_done = False

                    if idx == rl_idx:
                        # --- RL Agent turn ---
                        state  = get_state(player, engine, rl_idx)
                        action = select_action(state, rl.EPSILON)

                        # Track last decision for win bonus
                        last_state  = state
                        last_action = action

                        if action == "STOP":
                            leading_idx = engine.get_leading_opponent(rl_idx)
                            if leading_idx is not None:
                                score_gap = (engine.players[leading_idx].total_points
                                           - player.total_points)
                            else:
                                score_gap = 0

                            engine.bank_points(player)
                            reward = compute_reward(
                                points_banked=player.round_points,
                                busted=False,
                                score_gap=score_gap
                            )
                            next_state = get_state(player, engine, rl_idx)
                            update_q_table(state, action, reward,
                                          next_state, terminal=False)

                        elif action == "DRAW":
                            card = engine.draw_card()
                            engine.apply_card_by_index(rl_idx, card)

                            if player.busted:
                                player.stopped = True
                                leading_idx = engine.get_leading_opponent(rl_idx)
                                if leading_idx is not None:
                                    score_gap = (engine.players[leading_idx].total_points
                                               - player.total_points)
                                else:
                                    score_gap = 0

                                reward = compute_reward(
                                    points_banked=0,
                                    busted=True,
                                    score_gap=score_gap
                                )
                                update_q_table(state, action, reward,
                                             next_state=state,
                                             terminal=True)
                                stats["busts"] += 1
                            else:
                                next_state = get_state(player, engine, rl_idx)
                                update_q_table(state, action, reward=0.0,
                                             next_state=next_state,
                                             terminal=False)

                    else:
                        # --- Heuristic opponent turns ---
                        opponent_name = engine.players[idx].name
                        if opponent_name == "Conservative":
                            action = conservative_policy(player)
                        elif opponent_name == "Greedy":
                            action = greedy_policy(player)
                        elif opponent_name == "Balanced":
                            action = balanced_policy(player)

                        if action == "STOP":
                            engine.bank_points(player)
                        elif action == "DRAW":
                            card = engine.draw_card()
                            engine.apply_card_by_index(idx, card)
                            if player.busted:
                                player.stopped = True

                if all_done:
                    round_over = True

            # --- End of round ---
            # Bank frozen players
            for player in engine.players:
                if player.frozen and not player.busted:
                    player.total_points += (player.round_points
                                          * player.multiplier)

            # Proportional round reward
            agent        = engine.players[rl_idx]
            agent_score  = agent.round_points
            all_scores   = [p.round_points for p in engine.players
                           if not p.busted]

            if not agent.busted and all_scores and last_state is not None:
                if agent_score >= 15:
                    percentile   = (sorted(all_scores).index(agent_score)
                                  / len(all_scores))
                    round_bonus  = percentile * 0.3
                    if round_bonus > 0:
                        update_q_table(last_state, last_action,
                                      reward=round_bonus,
                                      next_state=last_state,
                                      terminal=False)
                    game_reward += round_bonus

            # Check win condition
            for player in engine.players:
                if player.total_points >= win_threshold:
                    if player.name == "RL_Agent":
                        stats["wins"] += 1
                        game_reward  += 5.0
                        if last_state is not None:
                            update_q_table(last_state, last_action,
                                          reward=5.0,
                                          next_state=last_state,
                                          terminal=True)
                    else:
                        stats["losses"] += 1
                    game_over = True
                    break

        # --- End of game ---
        recent_rewards.append(game_reward)
        if len(recent_rewards) > verbose_every:
            recent_rewards.pop(0)

        # Decay epsilon
        rl.EPSILON = max(EPSILON_MIN, rl.EPSILON * EPSILON_DECAY)

        # Snapshot logging
        if game_num % verbose_every == 0:
            win_rate   = stats["wins"] / game_num
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            stats["win_rate_log"].append(win_rate)
            stats["epsilon_log"].append(rl.EPSILON)
            stats["avg_reward_log"].append(avg_reward)
            stats["states_log"].append(len(q_table))

            log_training(
                f"Game {game_num:6d} | "
                f"Win Rate: {win_rate:.3f} | "
                f"Epsilon: {rl.EPSILON:.4f} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Busts: {stats['busts']} | "
                f"States: {len(q_table)}"
            )

    # Save Q-table after training
    save_q_table()
    log_training("Training complete.")
    return stats
