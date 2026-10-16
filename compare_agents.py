
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
BASE_SEED = 700

STATE_SIZE = 228
ACTION_SIZE = 200
HIDDEN_SIZE = 128

PROJECT_DIR = r"D:\ChefsHat-RL-Task2"
MODEL_PATH = os.path.join(PROJECT_DIR, "dqn_model.pth")
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "agent_comparison")
RESULTS_PATH = os.path.join(
    PROJECT_DIR, "agent_comparison_results.csv"
)

os.makedirs(LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = DQN().to(device)
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)
model.eval()


def play_match(condition, seed):
    # Seed the game's random-number generator.
    random.seed(seed)
    np.random.seed(seed)

    # Separate random choices by the agents from game randomness.
    agent_rng = random.Random(seed + 100_000)

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
    finished = False

    while steps < 1000:
        player = game.currentPlayer
        legal_mask = np.asarray(
            game.getPossibleActions(player),
            dtype=bool,
        )
        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            env.close()
            raise RuntimeError("No legal actions available.")

        # DQN is tested only as P0. Everyone else acts randomly.
        use_dqn = condition == "DQN_vs_random" and player == 0

        if use_dqn:
            state = torch.tensor(
                np.asarray(observation, dtype=np.float32),
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = model(state).squeeze(0).cpu().numpy()

            q_values[~legal_mask] = -np.inf
            action_index = int(np.argmax(q_values))
        else:
            action_index = agent_rng.choice(
                legal_indices.tolist()
            )

        action = np.zeros(ACTION_SIZE, dtype=np.float32)
        action[action_index] = 1.0

        with redirect_stdout(io.StringIO()):
            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

        steps += 1

        if terminated or truncated:
            finished = True
            break

    scores = [p.current_score for p in game.players]
    positions = [p.finishing_position for p in game.players]

    valid = (
        finished
        and steps < 1000
        and all(s >= 0 for s in scores)
        and all(p >= 0 for p in positions)
    )

    env.close()

    if not valid:
        return None

    return {
        "score": scores[0],
        "position": positions[0],
        "steps": steps,
    }


rows = []

for condition in ("DQN_vs_random", "Random_vs_random"):
    print(f"\nRunning condition: {condition}")

    for i in range(NUM_MATCHES):
        seed = BASE_SEED + i
        result = play_match(condition, seed)

        if result is None:
            print(f"Match {i + 1}: invalid; skipped")
            continue

        row = {
            "condition": condition,
            "match": i + 1,
            "seed": seed,
            **result,
        }
        rows.append(row)

        print(
            f"{i + 1}/{NUM_MATCHES} | "
            f"P0 score={result['score']} | "
            f"P0 position={result['position']}"
        )


with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition", "match", "seed",
            "score", "position", "steps",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


print("\n===== COMPARISON SUMMARY =====")
print("Results saved to:", RESULTS_PATH)

for condition in ("DQN_vs_random", "Random_vs_random"):
    subset = [r for r in rows if r["condition"] == condition]

    if not subset:
        print(condition, ": no valid results")
        continue

    average_score = np.mean([r["score"] for r in subset])
    win_rate = np.mean([
        r["position"] == 0 for r in subset
    ]) * 100

    print(
        f"{condition}: "
        f"{len(subset)} valid matches, "
        f"average P0 score={average_score:.2f}, "
        f"P0 first-place rate={win_rate:.1f}%"
    )
