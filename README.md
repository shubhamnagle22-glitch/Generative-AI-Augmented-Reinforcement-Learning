# Generative AI-Augmented Reinforcement Learning for Chef's Hat

## Assignment
- **Module:** Generative AI and Reinforcement Learning
- **Task 2 variant:** Variant 6 - Generative AI Augmentation
- **Environment:** Chef's Hat Gym
- **Language:** Python

## Project Overview

This project investigates whether a learned action-prior model can augment a Deep Q-Network (DQN) agent playing Chef's Hat.

The augmentation combines DQN action scores with learned action-prior scores. Legal-action masking prevents the agent from selecting illegal moves.

## Experiments

1. Random baseline: 30 matches.
2. DQN training and evaluation.
3. DQN self-play evaluation: 30 matches.
4. DQN versus random opponents, compared with a random-agent baseline: 30 matches per condition.
5. Learned action-prior augmentation: 30 matches.

## Results

| Agent / condition | Average score | First-place rate | Matches |
|---|---:|---:|---:|
| Random agent (P0) | 1.73 | 30.0% | 30 |
| DQN (P0) | 2.00 | 33.3% | 30 |
| Generative augmentation (P0) | 1.83 | 36.7% | 30 |

The augmentation condition achieved a higher first-place rate than the random baseline and the DQN evaluation in these runs. The DQN had the highest average score. These results are preliminary and do not establish statistical significance.

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
```

## Running the Project

Run the scripts from the project directory with the required dependencies installed:

```powershell
python random_baseline.py
python train_dqn.py
python evaluate_dqn.py
python compare_agents.py
python generative_augmentation.py
python plot_results.py
```

Some scripts train models or regenerate result files. Running them may overwrite existing outputs.

## Limitations

- Experiments use 30 matches per reported condition, so results may vary between runs.
- The results do not establish statistical significance.
- The action-prior model is a learned model trained from synthetic demonstrations; it is not a large language model.
- Performance may depend on training settings, random seeds, and the opponents used.

## AI Assistance Disclosure

AI assistance was used to support coding, debugging, and documentation. The project code and experimental results should be reviewed and understood by the student.
