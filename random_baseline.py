
import os
import random
import io
from contextlib import redirect_stdout

import gym
import numpy as np
import ChefsHatGym.env


NUM_MATCHES = 30
BASE_SEED = 42
LOG_DIR = r"D:\ChefsHat-RL-Task2\logs\random_baseline"

os.makedirs(LOG_DIR, exist_ok=True)

all_scores = []
first_place_counts = np.zeros(4, dtype=int)

for match_num in range(NUM_MATCHES):
    seed = BASE_SEED + match_num
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make("chefshat-v1")
    game = env.unwrapped

    # Start a fresh experiment for exactly one match.
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

    while steps < 1000:
        possible_actions = game.getPossibleActions(game.currentPlayer)
        legal_indices = np.flatnonzero(possible_actions)

        if len(legal_indices) == 0:
            raise RuntimeError(
                f"No legal actions in match {match_num + 1}"
            )

        # Choose a random legal action.
        action = np.zeros(200, dtype=np.float32)
        chosen_index = random.choice(legal_indices.tolist())
        action[chosen_index] = 1.0

        with redirect_stdout(io.StringIO()):
            observation, reward, terminated, truncated, info = env.step(action)

        steps += 1

        if terminated or truncated:
            break

    # Read results only after the single-match experiment finishes.
    scores = [player.current_score for player in game.players]
    positions = [player.finishing_position for player in game.players]

    if (
        steps >= 1000
        or any(score < 0 for score in scores)
        or any(position < 0 for position in positions)
    ):
        print(
            f"Match {match_num + 1}: invalid result "
            f"(steps={steps}, scores={scores}, positions={positions})"
        )
    else:
        all_scores.append(scores)

        for i, position in enumerate(positions):
            if position == 0:
                first_place_counts[i] += 1

        print(
            f"Match {match_num + 1}/{NUM_MATCHES}: "
            f"scores={scores}, steps={steps}"
        )

    env.close()

print("\n===== RANDOM AGENT BASELINE =====")
print(f"Valid matches: {len(all_scores)}/{NUM_MATCHES}")

if all_scores:
    score_array = np.array(all_scores, dtype=float)

    for i in range(4):
        print(
            f"P{i}: average score={score_array[:, i].mean():.2f}, "
            f"first-place rate="
            f"{first_place_counts[i] / len(all_scores) * 100:.1f}%"
        )
else:
    print("No valid matches recorded. Do not use these results.")
