# flip7 Evaluation File
# Evaluation suite — heuristic baselines, RL evaluation, statistical tests


import logging
import matplotlib.pyplot as plt
from numpy import random
from scipy import stats
from flip7_engine import Flip7RoundEngine, run_game
from flip7_policies import (conservative_policy, greedy_policy,
                             balanced_policy, conservative_agent,
                             greedy_agent, balanced_agent,
                             OPPONENT_CONFIGS)
from flip7_rl import q_table, get_state, select_action, load_q_table

# Evaluation logger
eval_logger = logging.getLogger("flip7.evaluation")

def log_eval(msg: str) -> None:
    eval_logger.info(msg)
    print(msg)


# ----------------------------
# Heuristic baseline — simple
# ----------------------------

def run_heuristic_baseline(num_games: int = 5000,
                           num_players: int = 4) -> dict:
    """
    Runs heuristic agents against each other.
    No RL agent involved.
    For purpose of report, using 4 to get a true baseline for the RL agent.
    num_players: 3 for Conservative/Greedy/Balanced,
                 4 adds RL_Agent slot filled by Balanced for fair comparison
    """
    if num_players == 4:
        player_names = ["Balanced_2", "Conservative", "Greedy", "Balanced"]
    else:
        player_names = ["Conservative", "Greedy", "Balanced"]

    policies = {}
    for i, name in enumerate(player_names):
        if "Conservative" in name:
            policies[i] = conservative_agent
        elif "Greedy" in name:
            policies[i] = greedy_agent
        else:
            policies[i] = balanced_agent

    results = {name: 0 for name in player_names}
    results["draws"] = 0

    for _ in range(num_games):
        result = run_game(
            player_names=player_names,
            policies=policies,
            win_threshold=200,
            verbose=False
        )
        winner = result["winner"]
        if winner in results:
            results[winner] += 1
        else:
            results["draws"] += 1

    log_eval(f"\nHeuristic Baseline Results over {num_games} games "
             f"({num_players} players):")
    for name in player_names:
        log_eval(f"  {name:<15} {results[name]/num_games:.3f} win rate")

    return results


# ----------------------------
# Heuristic baseline — detailed
# ----------------------------

def run_detailed_baseline(num_games: int = 5000) -> dict:
    """
    Tracks behavioral metrics per heuristic to diagnose performance.
    Runs 4-player config to match RL training conditions.
    """
    player_names = ["Balanced_2", "Conservative", "Greedy", "Balanced"]
    metrics = {
        name: {
            "wins": 0, "busts": 0, "total_banked": 0,
            "stops": 0, "rounds_played": 0
        }
        for name in player_names
    }

    for _ in range(num_games):
        engine   = Flip7RoundEngine(player_names)
        game_over = False

        while not game_over:
            engine.reset_round()

            while not engine.all_players_done():
                # Add in the run_roudn shuffle logic.
                turn_order = list(range(len(engine.players)))
                random.shuffle(turn_order)

                for idx in turn_order:
                    player = engine.players[idx]
                    if not player.is_active():
                        continue

                    name = player.name
                    metrics[name]["rounds_played"] += 1

                    if name == "Conservative":
                        action = conservative_policy(player)
                    elif name == "Greedy":
                        action = greedy_policy(player)
                    elif name in ["Balanced", "Balanced_2"]:
                        action = balanced_policy(player)

                    if action == "STOP":
                        metrics[name]["stops"]        += 1
                        metrics[name]["total_banked"] += player.round_points
                        engine.bank_points(player)
                    elif action == "DRAW":
                        card = engine.draw_card()
                        engine.apply_card_by_index(idx, card)
                        if player.busted:
                            metrics[name]["busts"] += 1
                            player.stopped = True

            # End of Round Logic
            for player in engine.players:
                if player.frozen and not player.busted:
                    player.total_points += (player.round_points
                                          * player.multiplier)
                    


            for player in engine.players:
                if player.total_points >= 200:
                    if player.name in metrics:
                        metrics[player.name]["wins"] += 1
                    game_over = True
                    break

    log_eval(f"\nDetailed Baseline Results over {num_games} games:")
    log_eval(f"{'Metric':<25} {'Conservative':>12} "
             f"{'Greedy':>12} {'Balanced':>12}")
    log_eval("-" * 63)

    for metric in ["wins", "busts", "stops", "total_banked"]:
        vals = {k: metrics[k][metric] for k in metrics}
        log_eval(f"{metric:<25} {vals['Conservative']:>12} "
                 f"{vals['Greedy']:>12} {vals['Balanced']:>12}")

    log_eval("\nDerived Metrics:")
    log_eval(f"{'Metric':<25} {'Conservative':>12} "
             f"{'Greedy':>12} {'Balanced':>12}")
    log_eval("-" * 63)

    for name in player_names:
        m          = metrics[name]
        avg_banked = m["total_banked"] / max(m["stops"], 1)
        bust_rate  = m["busts"]        / max(m["rounds_played"], 1)
        stop_rate  = m["stops"]        / max(m["rounds_played"], 1)
        log_eval(
            f"Avg banked: {avg_banked:>5.1f}  "
            f"Bust rate: {bust_rate:>5.3f}  "
            f"Stop rate: {stop_rate:>5.3f}  "
            f"— {name}"
        )

    return metrics


# ----------------------------
# RL agent evaluation
# ----------------------------

def eval_rl_vs_heuristics(num_games: int = 1000) -> dict:
    """
    Evaluates trained RL agent against heuristic opponents.
    Tracks win rate, avg points banked, bust rate, avg stopping score.
    Requires trained Q-table loaded via load_q_table().
    """
    player_names = OPPONENT_CONFIGS[3]
    rl_idx       = 0
    results = {
        "rl_wins":        0,
        "rl_busts":       0,
        "rl_total_banked": 0,
        "rl_stops":       0,
        "rl_rounds":      0,
        "opponent_wins":  0
    }

    for _ in range(num_games):
        engine    = Flip7RoundEngine(player_names)
        game_over = False

        while not game_over:
            engine.reset_round()

            while not engine.all_players_done():
                for idx, player in enumerate(engine.players):
                    if not player.is_active():
                        continue

                    if idx == rl_idx:
                        state  = get_state(player, engine, rl_idx)
                        action = select_action(state, epsilon=0.0)  # pure exploit

                        if action == "STOP":
                            results["rl_stops"]        += 1
                            results["rl_total_banked"] += player.round_points
                            engine.bank_points(player)
                        elif action == "DRAW":
                            card = engine.draw_card()
                            engine.apply_card_by_index(idx, card)
                            if player.busted:
                                results["rl_busts"] += 1
                                player.stopped = True
                    else:
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

            # end of round logic
            for player in engine.players:
                if player.frozen and not player.busted:
                    player.total_points += (player.round_points
                                          * player.multiplier)
                    
            results["rl_rounds"] += 1

            for player in engine.players:
                if player.total_points >= 200:
                    if player.name == "RL_Agent":
                        results["rl_wins"] += 1
                    else:
                        results["opponent_wins"] += 1
                    game_over = True
                    break

    win_rate   = results["rl_wins"]   / num_games
    bust_rate  = results["rl_busts"]  / max(results["rl_rounds"], 1)
    avg_banked = results["rl_total_banked"] / max(results["rl_stops"], 1)

    log_eval(f"\nRL Agent Evaluation over {num_games} games:")
    log_eval(f"  Win rate:        {win_rate:.3f}")
    log_eval(f"  Bust rate (per round):       {bust_rate:.3f}")
    log_eval(f"  Avg pts banked:  {avg_banked:.1f}")
    log_eval(f"  Opponent wins:   {results['opponent_wins']}")

    return results


# ----------------------------
# Statistical significance test
# ----------------------------

def run_statistical_tests(num_folds: int = 5,
                          games_per_fold: int = 200) -> dict:
    """
    5x2 cross-validation paired t-test comparing RL agent
    win rate against best heuristic baseline (Greedy).

    Required by rubric — reports t-statistic and p-value.
    H0: RL agent win rate == Greedy win rate
    H1: RL agent win rate != Greedy win rate
    """
    rl_scores      = []
    greedy_scores  = []

    for fold in range(num_folds):
        # RL agent performance
        rl_wins = 0
        for _ in range(games_per_fold):
            engine    = Flip7RoundEngine(OPPONENT_CONFIGS[3])
            game_over = False
            rl_idx    = 0

            while not game_over:
                engine.reset_round()

                while not engine.all_players_done():
                    for idx, player in enumerate(engine.players):
                        if not player.is_active():
                            continue

                        if idx == rl_idx:
                            state  = get_state(player, engine, rl_idx)
                            action = select_action(state, epsilon=0.0)
                            if action == "STOP":
                                engine.bank_points(player)
                            elif action == "DRAW":
                                card = engine.draw_card()
                                engine.apply_card_by_index(idx, card)
                                if player.busted:
                                    player.stopped = True
                        else:
                            opp = engine.players[idx].name
                            if opp == "Conservative":
                                action = conservative_policy(player)
                            elif opp == "Greedy":
                                action = greedy_policy(player)
                            elif opp == "Balanced":
                                action = balanced_policy(player)
                            if action == "STOP":
                                engine.bank_points(player)
                            elif action == "DRAW":
                                card = engine.draw_card()
                                engine.apply_card_by_index(idx, card)
                                if player.busted:
                                    player.stopped = True

                for player in engine.players:
                    if player.frozen and not player.busted:
                        player.total_points += (player.round_points
                                              * player.multiplier)

                for player in engine.players:
                    if player.total_points >= 200:
                        if player.name == "RL_Agent":
                            rl_wins += 1
                        game_over = True
                        break

        # Greedy agent performance in same config
        greedy_wins = 0
        policies = {
            0: greedy_agent,
            1: conservative_agent,
            2: balanced_agent,
            3: balanced_agent
        }
        for _ in range(games_per_fold):
            result = run_game(
                player_names=["Greedy", "Conservative",
                              "Balanced", "Balanced_2"],
                policies=policies,
                win_threshold=200,
                verbose=False
            )
            if result["winner"] == "Greedy":
                greedy_wins += 1

        rl_scores.append(rl_wins     / games_per_fold)
        greedy_scores.append(greedy_wins / games_per_fold)

        log_eval(f"Fold {fold+1}: RL={rl_wins/games_per_fold:.3f} "
                 f"Greedy={greedy_wins/games_per_fold:.3f}")

    t_stat, p_value = stats.ttest_rel(rl_scores, greedy_scores)

    log_eval(f"\nPaired t-test results:")
    log_eval(f"  H0: RL win rate == Greedy win rate")
    log_eval(f"  t-statistic: {t_stat:.4f}")
    log_eval(f"  p-value:     {p_value:.4f}")
    log_eval(f"  Significant (p<0.05): {p_value < 0.05}")

    return {
        "t_stat":       t_stat,
        "p_value":      p_value,
        "rl_scores":    rl_scores,
        "greedy_scores": greedy_scores
    }


# ----------------------------
# Learning curve plot
# ----------------------------

def plot_learning_curve(training_stats: dict,
                        save_path: str = "learning_curve.png") -> None:
    """
    Plots win rate and state coverage across training snapshots.
    Saves figure to disk for inclusion in report.
    """
    games    = [i * 1000 for i in range(1, len(
                training_stats["win_rate_log"]) + 1)]
    win_rate = training_stats["win_rate_log"]
    states   = training_stats["states_log"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(games, win_rate, marker="o", color="steelblue", linewidth=2)
    ax1.axhline(y=0.25, color="gray", linestyle="--",
                label="Random baseline (25%)")
    ax1.set_xlabel("Games Trained")
    ax1.set_ylabel("Win Rate")
    ax1.set_title("RL Agent Win Rate During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(games, states, marker="s", color="coral", linewidth=2)
    ax2.set_xlabel("Games Trained")
    ax2.set_ylabel("Unique States Visited")
    ax2.set_title("Q-Table State Coverage During Training")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    log_eval(f"Learning curve saved to {save_path}")
