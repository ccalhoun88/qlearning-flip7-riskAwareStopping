# flip7 Evaluation File
# Evaluation suite — heuristic baselines, RL evaluation, statistical tests


import logging
import matplotlib.pyplot as plt
import flip7_rl as rl
from numpy import random
from scipy import stats
from flip7_engine import Flip7RoundEngine, run_game
from flip7_policies import (conservative_policy, greedy_policy,
                             balanced_policy, conservative_agent,
                             greedy_agent, balanced_agent,
                             OPPONENT_CONFIGS)
from flip7_rl import get_state, select_action, load_q_table

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
        "opponent_wins":  0,
        "conservative_wins": 0,
        "greedy_wins":       0,
        "balanced_wins":      0
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
                    elif player.name == "Conservative":
                        results["conservative_wins"] += 1
                    elif player.name == "Greedy":
                        results["greedy_wins"] += 1
                    elif player.name == "Balanced":
                        results["balanced_wins"] += 1
                    results["opponent_wins"] += 1
                    game_over = True
                    break

    win_rate   = results["rl_wins"]   / num_games
    bust_rate  = results["rl_busts"]  / max(results["rl_rounds"], 1)
    avg_banked = results["rl_total_banked"] / max(results["rl_stops"], 1)

    log_eval(f"\nRL Agent Evaluation over {num_games} games:")
    log_eval(f"  {'Player':<15} {'Win Rate':>10}")
    log_eval(f"  {'-'*27}")
    log_eval(f"  {'RL_Agent':<15} {results['rl_wins']/num_games:>10.3f}")
    log_eval(f"  {'Conservative':<15} {results['conservative_wins']/num_games:>10.3f}")
    log_eval(f"  {'Greedy':<15} {results['greedy_wins']/num_games:>10.3f}")
    log_eval(f"  {'Balanced':<15} {results['balanced_wins']/num_games:>10.3f}")
    log_eval(f"  Bust rate (per round):       {bust_rate:>10.3f}")
    log_eval(f"  Avg pts banked:  {avg_banked:>10.1f}")

    return results


# ----------------------------
# Statistical significance test
# ----------------------------

def run_statistical_tests(num_folds: int = 5,
                          games_per_fold: int = 200) -> dict:
    """
    5.3.26 Update: Switching to K=10.
    Each fold retrains the RL agent from scratch on a different random seed, then evaluates on a separate set of games.
    Cross-validation paired t-test comparing RL agent
    win rate against best heuristic baseline (Greedy).
    Utilized Claude.ai to help update stat test logic.

    Required by rubric — reports t-statistic and p-value.
    H0: RL agent win rate == Greedy win rate
    H1: RL agent win rate != Greedy win rate
    """
    rl_scores      = []
    greedy_scores  = []
    conservative_scores = []
    balanced_scores     = []

    seeds = [42, 123, 456, 789, 1011,
             2024, 3141, 7777, 8888, 9999]

    for fold in range(num_folds):
        seed = seeds[fold]
        log_eval(f"\nFold {fold+1}/10 — Training with seed {seed}...")

        # Reset and retrain agent from scratch
        random.seed(seed)
        rl.q_table = defaultdict(lambda: {"DRAW": 0.0, "STOP": 0.0})
        rl.EPSILON = 1.0

        train_rl_agent(
            num_games=10000,
            num_opponents=3,
            win_threshold=200,
            verbose_every=10000  # only log final snapshot per fold
        )
      
        # Evaluate retrained RL agent performance
        rl_wins           = 0
        greedy_wins       = 0
        conservative_wins = 0
        balanced_wins     = 0
      
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
                        elif player.name == "Greedy":
                            greedy_wins += 1
                        elif player.name == "Conservative":
                            conservative_wins += 1
                        elif player.name == "Balanced":
                            balanced_wins += 1
                        game_over = True
                        break

        # Greedy Baseline Evaluation
        greedy_baseline_wins = 0
        for _ in range(games_per_fold):
            result = run_game(
                player_names=["Greedy", "Conservative",
                              "Balanced", "Balanced_2"],
                policies = {
                    0: greedy_agent,
                    1: conservative_agent,
                    2: balanced_agent,
                    3: balanced_agent
                },
                win_threshold=200,
                verbose=False
            )
            if result["winner"] == "Greedy":
                greedy_baseline_wins += 1

# Record fold results
        rl_rate           = rl_wins           / games_per_fold
        greedy_rate       = greedy_wins        / games_per_fold
        conservative_rate = conservative_wins  / games_per_fold
        balanced_rate     = balanced_wins      / games_per_fold
        greedy_base_rate  = greedy_baseline_wins / games_per_fold

        rl_scores.append(rl_rate)
        greedy_scores.append(greedy_base_rate)
        conservative_scores.append(conservative_rate)
        balanced_scores.append(balanced_rate)

        log_eval(f"Fold {fold+1} complete: "
                 f"RL={rl_rate:.3f} | "
                 f"Greedy={greedy_rate:.3f} | "
                 f"Conservative={conservative_rate:.3f} | "
                 f"Balanced={balanced_rate:.3f}")

    # --- Statistical tests ---
    t_stat,     p_value     = stats.ttest_rel(rl_scores, greedy_scores)
    t_stat_one, p_value_one = stats.ttest_rel(
        rl_scores, greedy_scores, alternative='less'
    )

    # Summary statistics
    mean_rl           = sum(rl_scores)           / num_folds
    mean_greedy       = sum(greedy_scores)        / num_folds
    mean_conservative = sum(conservative_scores)  / num_folds
    mean_balanced     = sum(balanced_scores)      / num_folds
    std_rl            = (sum((x - mean_rl)**2
                        for x in rl_scores) / num_folds) ** 0.5

    # 95% confidence interval
    ci = stats.t.interval(0.95,
                          df=num_folds - 1,
                          loc=mean_rl,
                          scale=stats.sem(rl_scores))

    log_eval(f"\n{'='*50}")
    log_eval(f"  STATISTICAL TEST RESULTS (k=10 folds)")
    log_eval(f"{'='*50}")
    log_eval(f"  H0: RL win rate == Greedy win rate")
    log_eval(f"  H1: RL win rate != Greedy win rate")
    log_eval(f"\n  Mean Win Rates across 10 folds:")
    log_eval(f"  {'Agent':<15} {'Mean Win Rate':>15}")
    log_eval(f"  {'-'*32}")
    log_eval(f"  {'RL Agent':<15} {mean_rl:>14.3f} +/- {std_rl:.3f}")
    log_eval(f"  {'Greedy':<15} {mean_greedy:>15.3f}")
    log_eval(f"  {'Conservative':<15} {mean_conservative:>15.3f}")
    log_eval(f"  {'Balanced':<15} {mean_balanced:>15.3f}")
    log_eval(f"\n  95% Confidence Interval: "
             f"({ci[0]:.3f}, {ci[1]:.3f})")
    log_eval(f"  t-statistic:             {t_stat:.4f}")
    log_eval(f"  p-value (two-sided):     {p_value:.4f}")
    log_eval(f"  p-value (one-sided):     {p_value_one:.4f}")
    log_eval(f"  Significant (p<0.05):    {p_value < 0.05}")

    return {
        "t_stat":             t_stat,
        "p_value":            p_value,
        "p_value_one":        p_value_one,
        "mean_rl":            mean_rl,
        "std_rl":             std_rl,
        "ci":                 ci,
        "mean_greedy":        mean_greedy,
        "mean_conservative":  mean_conservative,
        "mean_balanced":      mean_balanced,
        "rl_scores":          rl_scores,
        "greedy_scores":      greedy_scores,
        "conservative_scores": conservative_scores,
        "balanced_scores":    balanced_scores
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

# Bust Rate Comparison
# This is my design, but logic is from Claude.ai
def plot_bust_rate_comparison(eval_results: dict,
                               detailed_metrics: dict,
                               num_games: int = 1000,
                               save_path: str = "bust_rate_comparison.png") -> None:
    """
    Bar chart comparing bust rate per round across all players.
    RL agent bust rate from eval_rl_vs_heuristics.
    Heuristic bust rates from run_detailed_baseline.
    """
    players = ["RL Agent", "Conservative", "Greedy", "Balanced"]
    
    # RL bust rate already calculated per round
    rl_bust_rate = (eval_results["rl_busts"] / 
                   max(eval_results["rl_rounds"], 1))
    
    # Heuristic bust rates from detailed baseline metrics
    bust_rates = [
        rl_bust_rate,
        detailed_metrics["Conservative"]["busts"] / 
            max(detailed_metrics["Conservative"]["rounds_played"], 1),
        detailed_metrics["Greedy"]["busts"] / 
            max(detailed_metrics["Greedy"]["rounds_played"], 1),
        detailed_metrics["Balanced"]["busts"] / 
            max(detailed_metrics["Balanced"]["rounds_played"], 1)
    ]

    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(players, bust_rates, color=colors, 
                  width=0.5, edgecolor="white", linewidth=1.5)

    # Add value labels on top of each bar
    for bar, rate in zip(bars, bust_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{rate:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Player", fontsize=12)
    ax.set_ylabel("Bust Rate (per round)", fontsize=12)
    ax.set_title("Bust Rate Comparison — RL Agent vs Heuristics",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(bust_rates) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    log_eval(f"Bust rate comparison saved to {save_path}")


def plot_avg_points_banked(eval_results: dict,
                            detailed_metrics: dict,
                            save_path: str = "avg_points_banked.png") -> None:
    """
    Bar chart comparing average points banked per stop across all players.
    Shows the scoring efficiency gap between RL agent and heuristics.
    """
    players = ["RL Agent", "Conservative", "Greedy", "Balanced"]

    rl_avg_banked = (eval_results["rl_total_banked"] /
                    max(eval_results["rl_stops"], 1))

    avg_banked = [
        rl_avg_banked,
        detailed_metrics["Conservative"]["total_banked"] /
            max(detailed_metrics["Conservative"]["stops"], 1),
        detailed_metrics["Greedy"]["total_banked"] /
            max(detailed_metrics["Greedy"]["stops"], 1),
        detailed_metrics["Balanced"]["total_banked"] /
            max(detailed_metrics["Balanced"]["stops"], 1)
    ]

    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(players, avg_banked, color=colors,
                  width=0.5, edgecolor="white", linewidth=1.5)

    # Value labels on top of each bar
    for bar, val in zip(bars, avg_banked):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add reference line at RL agent's value for visual comparison
    ax.axhline(y=rl_avg_banked, color="steelblue", linestyle="--",
               alpha=0.5, label=f"RL Agent baseline ({rl_avg_banked:.1f})")

    ax.set_xlabel("Player", fontsize=12)
    ax.set_ylabel("Avg Points Banked Per Stop", fontsize=12)
    ax.set_title("Scoring Efficiency — Average Points Banked Per Stop",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(avg_banked) * 1.2)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    log_eval(f"Avg points banked chart saved to {save_path}")


def plot_q_value_heatmap(save_path: str = "q_value_heatmap.png") -> None:
    """
    Heatmap showing agent's preferred action by score bin and gap bin.
    Color represents preference margin — red=STOP, blue=DRAW.
    Only plots states the agent actually visited during training.
    """
    import numpy as np

    # Initialize grids for preference and margin
    # Rows = gap bins (0-3), Cols = score bins (0-7)
    preference_grid = np.zeros((4, 8))   # 1=STOP, -1=DRAW, 0=unknown
    margin_grid     = np.zeros((4, 8))   # absolute margin of preference

    for state, values in rl.q_table.items():
        score_bin, cards_drawn, gap_bin, has_second_chance = state

        # Only plot states with no second chance for clarity
        if has_second_chance:
            continue

        # Aggregate across cards_drawn by taking strongest signal
        draw_q = values["DRAW"]
        stop_q = values["STOP"]
        margin = abs(draw_q - stop_q)

        # Only update if this state has stronger signal than current
        if margin > margin_grid[gap_bin, score_bin]:
            margin_grid[gap_bin, score_bin]     = margin
            preference_grid[gap_bin, score_bin] = (
                1 if stop_q > draw_q else -1
            )

    # Build color grid — red=STOP, blue=DRAW, white=unknown
    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colormap — blue for DRAW, white for unknown, red for STOP
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "draw_stop",
        ["steelblue", "white", "crimson"]
    )

    im = ax.imshow(preference_grid * margin_grid,
                   cmap=cmap, aspect="auto",
                   vmin=-1.0, vmax=1.0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("← DRAW preference | STOP preference →",
                   fontsize=11)

    # Axis labels
    score_labels = [
        "0-12", "13-18", "19-23", "24-29",
        "30-36", "37-42", "43-49", "50+"
    ]
    gap_labels = [
        "Ahead\n(>15pts)",
        "Even\n(-15 to +15)",
        "Mod Behind\n(16-45pts)",
        "Far Behind\n(45+pts)"
    ]

    ax.set_xticks(range(8))
    ax.set_xticklabels(score_labels, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(gap_labels, fontsize=10)

    ax.set_xlabel("Round Score Bin", fontsize=12)
    ax.set_ylabel("Position vs Leading Opponent", fontsize=12)
    ax.set_title(
        "Q-Value Heatmap — Learned Action Preferences by State\n"
        "(Red=STOP, Blue=DRAW, White=Unvisited/Uncertain)",
        fontsize=13, fontweight="bold"
    )

    # Annotate cells with margin values
    for gap in range(4):
        for score in range(8):
            val = preference_grid[gap, score] * margin_grid[gap, score]
            if val != 0:
                label = f"{'S' if val > 0 else 'D'}\n{abs(val):.2f}"
                ax.text(score, gap, label,
                        ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(val) > 0.4 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    log_eval(f"Q-value heatmap saved to {save_path}")
