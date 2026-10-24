import os

import gym
import ChefsHatGym.env  # Registers chefshat-v1

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "smoke_test")
os.makedirs(LOG_DIR, exist_ok=True)


env = gym.make("chefshat-v1")
game = env.unwrapped

try:
    # Configure four players and initialise the experiment.
    game.startExperiment(
        gameType="POINTS",
        stopCriteria=1,
        playerNames=["P0", "P1", "P2", "P3"],
        logDirectory=LOG_DIR,
        verbose=0,
        saveLog=False,
        saveDataset=False,
    )

    observation, info = env.reset(seed=42)

    print("Environment initialised successfully!")
    print("Observation shape:", observation.shape)
    print("Observation data type:", observation.dtype)
    print("Initial observation:", observation)
    print("Reset info:", info)

finally:
    env.close()
