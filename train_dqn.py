
import os
import random
import io
import csv
from collections import deque
from contextlib import redirect_stdout

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ChefsHatGym.env


# ============================================================
# CONFIGURATION
# ============================================================
NUM_MATCHES = 30
BASE_SEED = 100

STATE_SIZE = 228
ACTION_SIZE = 200
HIDDEN_SIZE = 128

GAMMA = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000

EPSILON_START = 1.0
EPSILON_MIN = 0.10
EPSILON_DECAY = 0.97

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "dqn_training")
MODEL_PATH = os.path.join(PROJECT_DIR, "dqn_model.pth")
RESULTS_PATH = os.path.join(
    PROJECT_DIR, "dqn_training_results.csv"
)

os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DQN NETWORK
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
# REPLAY MEMORY
# Each item:
# state, action, reward, next_state, done, next_legal_mask
# ============================================================
memory = deque(maxlen=MEMORY_SIZE)


def train_step(policy, target, optimizer):
    if len(memory) < BATCH_SIZE:
        return None

    batch = random.sample(memory, BATCH_SIZE)

    states, actions, rewards, next_states, dones, next_masks = zip(
        *batch
    )

    states = torch.tensor(
        np.asarray(states), dtype=torch.float32, device=device
    )
    actions = torch.tensor(
        actions, dtype=torch.long, device=device
    )
    rewards = torch.tensor(
        rewards, dtype=torch.float32, device=device
    )
    next_states = torch.tensor(
        np.asarray(next_states), dtype=torch.float32, device=device
    )
    dones = torch.tensor(
        dones, dtype=torch.float32, device=device
    )
    next_masks = torch.tensor(
        np.asarray(next_masks), dtype=torch.bool, device=device
    )

    current_q = policy(states).gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)

    with torch.no_grad():
        next_q = target(next_states)

        # Ignore illegal next actions.
        next_q[~next_masks] = -1e9

        best_next_q = next_q.max(dim=1).values

        targets = rewards + GAMMA * best_next_q * (1 - dones)

    loss = nn.functional.smooth_l1_loss(
        current_q, targets
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        policy.parameters(), max_norm=1.0
    )

    optimizer.step()

    return float(loss.item())


# ============================================================
# ENVIRONMENT HELPER
# ============================================================
def start_single_match(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    return env, game, observation, info


# ============================================================
# INITIALIZE DQN
# ============================================================
policy = DQN().to(device)
target = DQN().to(device)

target.load_state_dict(policy.state_dict())
target.eval()

optimizer = optim.Adam(
    policy.parameters(), lr=LEARNING_RATE
)

epsilon = EPSILON_START
results = []
losses = []

print("Starting DQN training")
print("Device:", device)
print("Training matches:", NUM_MATCHES)


# ============================================================
# TRAINING LOOP
# ============================================================
for match_num in range(NUM_MATCHES):
    seed = BASE_SEED + match_num

    env, game, observation, info = start_single_match(seed)

    # Keep a separate sequence of decisions for each player.
    player_histories = {i: [] for i in range(4)}

    steps = 0
    match_finished = False

    while steps < 1000:
        acting_player = game.currentPlayer

        legal_mask = np.asarray(
            game.getPossibleActions(acting_player),
            dtype=bool,
        )

        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            raise RuntimeError(
                f"No legal actions in match {match_num + 1}"
            )

        state = np.asarray(
            observation, dtype=np.float32
        ).copy()

        # Epsilon-greedy selection among legal actions.
        if random.random() < epsilon:
            action_index = random.choice(
                legal_indices.tolist()
            )
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                q_values = policy(
                    state_tensor
                ).squeeze(0).cpu().numpy()

            q_values[~legal_mask] = -np.inf
            action_index = int(np.argmax(q_values))

        action = np.zeros(
            ACTION_SIZE, dtype=np.float32
        )
        action[action_index] = 1.0

        with redirect_stdout(io.StringIO()):
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

        # Save this player's decision state and action.
        player_histories[acting_player].append(
            (state, action_index, legal_mask)
        )

        observation = next_observation
        steps += 1

        if terminated or truncated:
            match_finished = True
            break

    # Read final match outcome.
    scores = [
        player.current_score for player in game.players
    ]
    positions = [
        player.finishing_position for player in game.players
    ]

    valid_result = (
        match_finished
        and steps < 1000
        and all(score >= 0 for score in scores)
        and all(position >= 0 for position in positions)
    )

    if not valid_result:
        print(
            f"Match {match_num + 1}: invalid result; skipped."
        )
        env.close()
        continue

    # --------------------------------------------------------
    # Create transitions for each player's own decisions.
    #
    # Intermediate turns have reward 0 and bootstrap from
    # that same player's next decision state.
    #
    # The final decision receives the player's match score
    # as reward and is marked terminal.
    # --------------------------------------------------------
    for player_id in range(4):
        trajectory = player_histories[player_id]

        for t, (state, action_index, action_mask) in enumerate(
            trajectory
        ):
            is_final_turn = (
                t == len(trajectory) - 1
            )

            if is_final_turn:
                next_state = np.zeros(
                    STATE_SIZE, dtype=np.float32
                )
                next_mask = np.zeros(
                    ACTION_SIZE, dtype=bool
                )
                outcome_reward = float(
                    3 - positions[player_id]
                )
                done = True
            else:
                next_state, _, next_mask = (
                    trajectory[t + 1]
                )
                outcome_reward = 0.0
                done = False

            memory.append(
                (
                    state,
                    action_index,
                    outcome_reward,
                    next_state,
                    done,
                    next_mask,
                )
            )

    # Learn from replay memory.
    for _ in range(min(steps, 20)):
        loss = train_step(
            policy, target, optimizer
        )

        if loss is not None:
            losses.append(loss)

    # Update the target network periodically.
    if (match_num + 1) % 5 == 0:
        target.load_state_dict(
            policy.state_dict()
        )

    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY,
    )

    results.append({
        "match": match_num + 1,
        "steps": steps,
        "P0_score": scores[0],
        "P1_score": scores[1],
        "P2_score": scores[2],
        "P3_score": scores[3],
        "P0_position": positions[0],
        "P1_position": positions[1],
        "P2_position": positions[2],
        "P3_position": positions[3],
        "epsilon": epsilon,
    })

    print(
        f"Match {match_num + 1}/{NUM_MATCHES} | "
        f"steps={steps} | scores={scores} | "
        f"epsilon={epsilon:.3f} | "
        f"memory={len(memory)}"
    )

    env.close()


# ============================================================
# SAVE MODEL AND RESULTS
# ============================================================
torch.save(
    policy.state_dict(), MODEL_PATH
)

fieldnames = [
    "match",
    "steps",
    "P0_score",
    "P1_score",
    "P2_score",
    "P3_score",
    "P0_position",
    "P1_position",
    "P2_position",
    "P3_position",
    "epsilon",
]

with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file, fieldnames=fieldnames
    )
    writer.writeheader()
    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================
print("\n===== DQN TRAINING SUMMARY =====")
print(
    f"Valid matches: {len(results)}/{NUM_MATCHES}"
)
print("Model saved to:", MODEL_PATH)
print("Results saved to:", RESULTS_PATH)

if results:
    for player in range(4):
        average_score = np.mean([
            row[f"P{player}_score"]
            for row in results
        ])

        first_place_rate = np.mean([
            row[f"P{player}_position"] == 0
            for row in results
        ]) * 100

        print(
            f"P{player}: average score="
            f"{average_score:.2f}, "
            f"first-place rate="
            f"{first_place_rate:.1f}%"
        )

if losses:
    print(
        f"Average training loss: "
        f"{np.mean(losses):.4f}"
    )
else:
    print(
        "No training loss recorded. "
        "Check replay memory and match results."
    )
