# Learning Risk-Aware Policies with Flip 7
## A Comparative Study of Reinforcement Learning and Heuristic Agents

* Diante Calhoun
* CIS 730 Artificial Intelligence | Dr. William Hsu | Spring 2026
* Kansas State University

## Project Overview
This Project investigates whether a tabular Q-learning agent can discover risk-aware, position-sensitive stopping politices in the stochastic card game Flip 7.
For purposes of theis project, three heuristiic baseline agents (Conservative, Greedy, and Balanced) are compared against a trained RL agent across 10,000 simulated games.

Full methodology and results are documented in the accompanying Term paper.



## Repository Structure  
flip7/  
├── DcalhounFlip7.ipynb       # My original python notebook used to review and continuously adjust the heuristics, game engine and the RL Agent  
├── flip7_engine.py       # Core game — deck, PlayerState, Flip7RoundEngine  
├── flip7_policies.py     # Heuristic agents and opponent configurations  
├── flip7_rl.py           # Q-table, state representation, reward function  
├── flip7_training.py     # Training loop and logging  
├── flip7_evaluation.py   # Baselines, RL evaluation, statistical tests  
├── main.py               # Entry point — runs full pipeline  
├── q_table.pkl           # Trained Q-table (skip retraining)  
├── learning_curve.png    # Win rate and state coverage figure  
├── logs/  
│   ├── flip7_game_log.txt      # Game events during training  
│   ├── flip7_training_log.txt  # Win rate snapshots every 1000 games  
│   └── flip7_eval_log.txt      # Evaluation and statistical test results  
├── README.md  
└── requirements.txt  


---

## Setup Instructions

### 1. Prerequisites
- Python 3.11.7
- pip

### 2. Clone the repository
```bash
git clone https://github.com/ccalhoun88/qlearning-flip7-riskAwareStopping.git
cd qlearning-flip7-riskAwareStopping
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python main.py
```

This will:
- Run 3-player and 4-player heuristic baselines
- Load the saved Q-table from `q_table.pkl` (skips retraining)
- Evaluate the RL agent over 1,000 games
- Run paired t-test vs Greedy baseline
- Print results summary to console
- Save all logs to `/logs` directory

### 5. Retrain from scratch (optional)
To retrain the agent, delete `q_table.pkl` and re-run:
```bash
del q_table.pkl      # Windows
rm q_table.pkl       # Mac/Linux
python main.py
```

Training runs 10,000 games and takes approximately 3-5 minutes.

---

## Key Results

| Agent         | Win Rate | Avg Points Banked | Bust Rate |
|---------------|----------|-------------------|-----------|
| Conservative  | 0.266    | 21.6              | 0.046     |
| Greedy        | 0.327    | 34.0              | 0.090     |
| Balanced      | 0.254    | 29.7              | 0.076     |
| RL Agent      | 0.153    | 18.4              | 0.110     |

*4-player baseline results — update after running main.py*

**Statistical Test:** Paired t-test, RL Agent vs Greedy
- t-statistic: TBD
- p-value: TBD

---

## Reproducing Exact Results

To reproduce the exact training results from the report, use the
saved Q-table included in the repository (`q_table.pkl`).

For a seeded training run:
```python
from flip7_engine import Flip7RoundEngine
engine = Flip7RoundEngine(player_names, seed=42)
```

---

## GenAI Usage

This project used Claude (Anthropic), ChatGPT 5.5, and VSCode Gemini Copilot as a development assistant for:
- Initial scaffolding of the game simulator – ChatGPT
	Original Prompt:
Create a lightweight custom Python simulation environment for the stochastic card game Flip 7 to support future reinforcement learning experiments using tabular Q-learning. Use simple card representations (integers and strings) instead of complex object-oriented card classes. The environment should include deck creation, player state tracking, round scoring, duplicate-card bust logic, special card handling (Freeze, Draw 3, Second Chance, multipliers, and bonus cards), heuristic-driven opponent behaviors (Conservative, Greedy, and Balanced), and multiplayer round flow for 2–4 players. Implement the environment using a Gym-style structure with reset(), step(), and state representation concepts, but avoid requiring OpenAI Gym itself. The initial focus should be on round-level training episodes rather than full-game simulations to simplify reward assignment and accelerate reinforcement learning experimentation
- Debugging training loop logic 
- Guidance on Q-learning architecture decisions 
- Debugging Q-learning State Space
- Debugging Heuristic Baseline Evaluation logic
- Conceptual guidance on Q-Learning Design Decisions
- LaTex and Overleaf formatting
 - Report Writing Refinement
All technical design decisions, hyperparameter choices, reward function design, and analytical conclusions are my own work.
In regards to writing refinement, Claude was used throughout the report drafting process with an initial draft, follow up targeted questions from Claude, which was then refined and into IEEE-formatted academic prose. All technical content, design decisions, analytical conclusions and observations originate from my own work and understanding.


---

## Dependencies

- Python 3.11.7
- scipy >= 1.11
- matplotlib >= 3.7
- numpy >= 1.24

Standard library: random, logging, pickle, collections, dataclasses
