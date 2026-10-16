
import os
import io
import csv
import random
from contextlib import redirect_stdout

import gym
import numpy as np
import torch
import torch.nn as nn
import ChefsHatGym.env


# Configuration
NUM_MATCHES = 30
BASE_SEED = 500

STATE_SIZE = 228
ACTION_SIZE = 200
HIDDEN_SIZE = 128

PROJECT_DIR = r"D:\ChefsHat-RL-Task2"
MODEL_PATH = os.path.join(PROJECT_DIR, "dqn_model.pth")
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "dqn_evaluation")
RESULTS_PATH = os.path.join(
    PROJECT_DIR, "evaluation_results.csv"
)

os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, ACTION_SIZE),
        )

    def forward(self, state):
        return self.network(state)


# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run train_dqn.py first."
    )

model = DQN().to(device)
state_dict = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

print("Loaded DQN model:", MODEL_PATH)
print("Device:", device)
print("Evaluation matches:", NUM_MATCHES)


results = []

for match_num in range(NUM_MATCHES):
    seed = BASE_SEED + match_num
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make("chefshat-v1")
    game = env.unwrapped

    with redirect_stdout(io.StringIO()):
        game.startExperiment(
            gameType="POINTS",
            stopCriteria=1,
            playerNames=["P0", "P1", "P2", "P3"],
            logDirectory=LOG_DIR,
            verbose=0,
            saveLog=False,
            saveDataset=False,
        )
        observation, info = env.reset(seed=seed)

    steps = 0
    terminated = False

    while steps < 1000:
        player_index = game.currentPlayer

        legal_mask = np.asarray(
            game.getPossibleActions(player_index),
            dtype=bool,
        )
        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            raise RuntimeError(
                f"No legal actions in match {match_num + 1}"
            )

        state = torch.tensor(
            np.asarray(observation, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        # Greedy policy: no random exploration.
        with torch.no_grad():
            q_values = model(state).squeeze(0).cpu().numpy()

        q_values[~legal_mask] = -np.inf
        action_index = int(np.argmax(q_values))

        action = np.zeros(ACTION_SIZE, dtype=np.float32)
        action[action_index] = 1.0

        with redirect_stdout(io.StringIO()):
            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

        steps += 1

        if terminated or truncated:
            break

    scores = [player.current_score for player in game.players]
    positions = [
        player.finishing_position for player in game.players
    ]

    valid = (
        (terminated or truncated)
        and steps < 1000
        and all(score >= 0 for score in scores)
        and all(position >= 0 for position in positions)
    )

    if valid:
        row = {
            "match": match_num + 1,
            "steps": steps,
        }

        for i in range(4):
            row[f"P{i}_score"] = scores[i]
            row[f"P{i}_position"] = positions[i]

        results.append(row)

        print(
            f"Match {match_num + 1}/{NUM_MATCHES} | "
            f"scores={scores} | positions={positions}"
        )
    else:
        print(
            f"Match {match_num + 1}/{NUM_MATCHES}: "
            "invalid result; skipped."
        )

    env.close()


# Save results
fieldnames = [
    "match", "steps",
    "P0_score", "P1_score", "P2_score", "P3_score",
    "P0_position", "P1_position", "P2_position", "P3_position",
]

with open(
    RESULTS_PATH, "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)


# Summarize performance
print("\n===== DQN EVALUATION SUMMARY =====")
print(f"Valid matches: {len(results)}/{NUM_MATCHES}")
print("Results saved to:", RESULTS_PATH)

if results:
    for i in range(4):
        average_score = np.mean([
            row[f"P{i}_score"] for row in results
        ])
        first_place_rate = np.mean([
            row[f"P{i}_position"] == 0 for row in results
        ]) * 100

        print(
            f"P{i}: average score={average_score:.2f}, "
            f"first-place rate={first_place_rate:.1f}%"
        )
else:
    print("No valid matches. Do not draw conclusions yet.")

