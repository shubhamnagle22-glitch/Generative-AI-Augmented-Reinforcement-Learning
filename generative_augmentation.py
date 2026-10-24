
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


# ============================================================
# CONFIGURATION
# ============================================================
NUM_MATCHES = 30
BASE_SEED = 900

STATE_SIZE = 228
ACTION_SIZE = 200
HIDDEN_SIZE = 128

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "dqn_model.pth")
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "generative_augmentation")
RESULTS_PATH = os.path.join(
    PROJECT_DIR, "generative_augmentation_results.csv"
)

os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DQN MODEL
# ============================================================
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


# ============================================================
# GENERATIVE ACTION-PRIOR MODEL
# Conditional generator learns a distribution over action
# preferences from legal-action demonstrations.
# ============================================================
class ActionPriorGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE),
        )

    def forward(self, state):
        return self.network(state)


def build_generator():
    generator = ActionPriorGenerator().to(device)
    return generator


def create_synthetic_demonstrations(num_samples=5000, seed=1234):
    """
    Create legal-action demonstrations from random play.
    The demonstrations provide a baseline action distribution;
    they are not expert demonstrations.
    """
    rng = random.Random(seed)
    states = []
    action_labels = []

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

    while len(states) < num_samples:
        legal_mask = np.asarray(
            game.getPossibleActions(game.currentPlayer),
            dtype=bool,
        )
        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            break

        action_index = rng.choice(legal_indices.tolist())

        states.append(
            np.asarray(observation, dtype=np.float32).copy()
        )
        action_labels.append(action_index)

        action = np.zeros(ACTION_SIZE, dtype=np.float32)
        action[action_index] = 1.0

        with redirect_stdout(io.StringIO()):
            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

        if terminated or truncated:
            with redirect_stdout(io.StringIO()):
                observation, info = env.reset()

    env.close()

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(action_labels, dtype=np.int64),
    )


def train_generator(generator, states, labels, epochs=5):
    """
    Train the generator to model action choices from synthetic
    random-agent demonstrations. This is a generative-style
    action-prior component, not a pretrained LLM.
    """
    dataset_size = len(states)
    if dataset_size == 0:
        raise RuntimeError("No demonstration data collected.")

    optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.001
    )
    loss_function = nn.CrossEntropyLoss()

    x = torch.tensor(states, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size, device=device)
        total_loss = 0.0
        batches = 0

        for start in range(0, dataset_size, 64):
            indices = permutation[start:start + 64]
            logits = generator(x[indices])
            loss = loss_function(logits, y[indices])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        print(
            f"Generator epoch {epoch + 1}/{epochs} | "
            f"loss={total_loss / max(batches, 1):.4f}"
        )

    return generator


# ============================================================
# LOAD DQN AND TRAIN GENERATIVE ACTION PRIOR
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"DQN model not found: {MODEL_PATH}"
    )

dqn = DQN().to(device)
dqn.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)
dqn.eval()

print("Collecting synthetic legal-action demonstrations...")
demo_states, demo_labels = create_synthetic_demonstrations()

print("Demonstrations collected:", len(demo_states))

generator = build_generator()
generator = train_generator(
    generator, demo_states, demo_labels, epochs=5
)

GENERATOR_PATH = os.path.join(
    PROJECT_DIR, "action_prior_generator.pth"
)
torch.save(generator.state_dict(), GENERATOR_PATH)
generator.eval()

print("Generator saved to:", GENERATOR_PATH)


# ============================================================
# EVALUATION: DQN + GENERATIVE ACTION PRIOR VS RANDOM
# ============================================================
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
    finished = False

    while steps < 1000:
        player_index = game.currentPlayer
        legal_mask = np.asarray(
            game.getPossibleActions(player_index),
            dtype=bool,
        )
        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available.")

        if player_index == 0:
            state_tensor = torch.tensor(
                np.asarray(observation, dtype=np.float32),
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                dqn_scores = dqn(state_tensor).squeeze(0)
                generated_logits = generator(
                    state_tensor
                ).squeeze(0)

                # Normalize scores before combining.
                dqn_scores = (
                    dqn_scores - dqn_scores.mean()
                ) / (dqn_scores.std() + 1e-6)

                generated_logits = (
                    generated_logits - generated_logits.mean()
                ) / (generated_logits.std() + 1e-6)

                # Combined policy: DQN score plus a generative
                # action-prior preference.
                combined_scores = (
                    0.8 * dqn_scores + 0.2 * generated_logits
                ).cpu().numpy()

            combined_scores[~legal_mask] = -np.inf
            action_index = int(np.argmax(combined_scores))
        else:
            action_index = random.choice(
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
        and all(score >= 0 for score in scores)
        and all(position >= 0 for position in positions)
    )

    if valid:
        results.append({
            "match": match_num + 1,
            "seed": seed,
            "steps": steps,
            "P0_score": scores[0],
            "P0_position": positions[0],
            "P1_score": scores[1],
            "P2_score": scores[2],
            "P3_score": scores[3],
        })

        print(
            f"Match {match_num + 1}/{NUM_MATCHES} | "
            f"P0 score={scores[0]} | "
            f"P0 position={positions[0]}"
        )
    else:
        print(f"Match {match_num + 1}: invalid result; skipped.")

    env.close()


# ============================================================
# SAVE RESULTS
# ============================================================
with open(
    RESULTS_PATH, "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "match", "seed", "steps",
            "P0_score", "P0_position",
            "P1_score", "P2_score", "P3_score",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print("\n===== GENERATIVE AUGMENTATION RESULTS =====")
print(f"Valid matches: {len(results)}/{NUM_MATCHES}")
print("Results saved to:", RESULTS_PATH)

if results:
    print(
        "P0 average score:",
        round(np.mean([r["P0_score"] for r in results]), 3),
    )
    print(
        "P0 first-place rate:",
        round(
            np.mean([
                r["P0_position"] == 0 for r in results
            ]) * 100,
            1,
        ),
        "%",
    )
