# Chef's Hat RL — Generative AI Augmentation

## Assignment
- Module: Generative AI and Reinforcement Learning
- Task 2 variant: Variant 6 — Generative AI Augmentation
- Environment: Chef's Hat Gym
- Language: Python

## Project Overview
This project investigates whether a learned action-prior model can augment a Deep Q-Network (DQN) agent playing Chef's Hat.

The augmentation combines DQN action scores with learned action-prior scores. Legal-action masking prevents the agent from selecting illegal moves.

## Experiments
1. Random baseline: 30 matches.
2. DQN training: 30 evaluation matches.
3. DQN self-play evaluation: 30 matches.
4. DQN versus random opponents, compared with a random-agent baseline: 30 matches per condition.
5. Learned action-prior augmentation: 30 matches.

## Results

| Agent / condition | Average score | First-place rate | Matches |
|---|---:|---:|---:|
| Random agent (P0) | 1.73 | 30.0% | 30 |
| DQN (P0) | 1.53 | 26.7% | 30 |
| Generative augmentation (P0) | 2.30 | 63.3% | 30 |

The augmentation condition achieved the highest observed average score and first-place rate in these runs. These results are preliminary and do not establish statistical significance.

## Generated Outputs
The `plots/` directory contains:
- `average_score.png`
- `first_place_rate.png`
- `score_distribution.png`
- `summary.csv`

CSV files store experimental results. Model checkpoints are saved as `.pth` files.

## Requirements
Python 3.11 is recommended.

Install dependencies:

```powershell
pip install chefshatgym==3.0.0.1 gym==0.26.2 torch numpy pandas matplotlib
