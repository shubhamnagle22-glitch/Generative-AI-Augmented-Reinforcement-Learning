import gym
from pathlib import Path
import numpy as np
import ChefsHatGym.env

env = gym.make("chefshat-v1")
game = env.unwrapped

game.startExperiment(
    gameType="POINTS",
    stopCriteria=1,
    playerNames=["P0", "P1", "P2", "P3"],
    logDirectory=str(Path(__file__).resolve().parent / "logs" / "test_match"),
    verbose=0,
    saveLog=False,
    saveDataset=False,
)

observation, info = env.reset(seed=42)

steps = 0
max_steps = 5000

while steps < max_steps:
    player = game.currentPlayer
    legal_actions = np.flatnonzero(
        game.getPossibleActions(player)
    )

    action = np.zeros(200, dtype=np.float32)
    action[int(np.random.choice(legal_actions))] = 1.0

    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if terminated or truncated:
        break

print("Steps:", steps)
print("Match over:", terminated)
print("Last reward:", reward)
print("Match score:", info.get("Match_Score"))
print("Accumulated score:", info.get("Game_Score"))
print(
    "Finishing positions:",
    [p.finishing_position for p in game.players],
)

env.close()