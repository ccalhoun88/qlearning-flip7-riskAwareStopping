# Added logging for the two loggers in the engine and the training
import logging
import flip7_rl as rl
import os
from flip7_engine import Flip7RoundEngine
from flip7_policies import OPPONENT_CONFIGS
from flip7_rl import load_q_table, save_q_table
from flip7_training import train_rl_agent
from flip7_evaluation import (run_heuristic_baseline,
                               run_detailed_baseline,
                               eval_rl_vs_heuristics,
                               run_statistical_tests,
                               plot_learning_curve,
                               plot_bust_rate_comparison,
                               plot_avg_points_banked,
                               plot_q_value_heatmap)


def setup_logging():
    """
    Configures named loggers for engine and training.
    Called once at startup before any other imports run logic.
    """
    os.makedirs("logs", exist_ok=True) # This creates the log subdirectory instead of creating everytime.
    # Engine game events
    engine_handler = logging.FileHandler("flip7_game_log.txt", mode="w")
    engine_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("flip7.engine").addHandler(engine_handler)
    logging.getLogger("flip7.engine").setLevel(logging.INFO)

    # Training metrics
    training_handler = logging.FileHandler("flip7_training_log.txt", mode="w")
    training_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("flip7.training").addHandler(training_handler)
    logging.getLogger("flip7.training").setLevel(logging.INFO)

    # Evaluation results
    eval_handler = logging.FileHandler("logs/flip7_eval_log.txt", mode="w")
    eval_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("flip7.evaluation").addHandler(eval_handler)
    logging.getLogger("flip7.evaluation").setLevel(logging.INFO)

    print("Logging configured — outputs in /logs directory")


#---------------------
# Main Pipeline
#---------------------

def main():
    setup_logging()

    print("\n" + "="*50)
    print("  FLIP 7 — RL AGENT TRAINING & EVALUATION")
    print("="*50)

    # Heuristic Baselines
    print("\n[Step 1] Running heuristic baselines...")
    detailed_metrics = run_detailed_baseline(num_games=5000)
    run_heuristic_baseline(num_games=5000, num_players=4)


    # Train or load agent
    # This is the additional add from Claude.ai that is utilizing the files saved from q-table.
    print("\n[Step 2] Checking for saved Q-table...")

    if os.path.exists("q_table.pkl"):
        print("Found saved Q-table — loading...")
        load_q_table()
        print(f"Q-table loaded: {len(rl.q_table)} states")
    else:
        print("No saved Q-table found — training from scratch...")
        training_stats = train_rl_agent(
            num_games=10000,
            num_opponents=3,
            win_threshold=200,
            verbose_every=1000
        )
        plot_learning_curve(training_stats)
    
    print(f"Q-table states after training: {len(rl.q_table)}")


    # Evaluate RL agent
    print("\n[Step 3] Evaluating RL agent...")
    eval_results = eval_rl_vs_heuristics(num_games=1000)


    # Statistical Tests
    print("\n[Step 4] Running statistical significance tests...")
    test_results = run_statistical_tests(
        num_folds=5,
        games_per_fold=200
    )


    # Summary
    print(f"\n{'='*50}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"RL Agent win rate:         {eval_results['rl_wins']/1000:.3f}")
    print(f"RL bust rate/round:        "
          f"{eval_results['rl_busts']/max(eval_results['rl_rounds'],1):.3f}")
    print(f"RL avg pts banked:         "
          f"{eval_results['rl_total_banked']/max(eval_results['rl_stops'],1):.1f}")
    print(f"\nStatistical Test (k=10 folds):")
    print(f"RL mean win rate:          "
          f"{test_results['mean_rl']:.3f} +/- {test_results['std_rl']:.3f}")
    print(f"Greedy mean win rate:      {test_results['mean_greedy']:.3f}")
    print(f"Conservative mean:         {test_results['mean_conservative']:.3f}")
    print(f"Balanced mean:             {test_results['mean_balanced']:.3f}")
    print(f"95% CI:                    "
          f"({test_results['ci'][0]:.3f}, {test_results['ci'][1]:.3f})")
    print(f"t-statistic:               {test_results['t_stat']:.4f}")
    print(f"p-value (two-sided):       {test_results['p_value']:.4f}")
    print(f"p-value (one-sided):       {test_results['p_value_one']:.4f}")
    print(f"Significant (p<0.05):      {test_results['p_value'] < 0.05}")
    print(f"\nAll logs saved to /logs directory")
    print(f"Q-table saved to q_table.pkl")
    print(f"Learning curve saved to learning_curve.png")

    # Generating graphs
    print("\n[Step 6] Generating images...")
    plot_bust_rate_comparison(eval_results, detailed_metrics)
    plot_avg_points_banked(eval_results, detailed_metrics)
    plot_q_value_heatmap()          
    print("All figures saved.")

if __name__ == "__main__":
    main()

