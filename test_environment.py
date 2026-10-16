
import gym
import ChefsHatGym.env  # Registers chefshat-v1

env = gym.make("chefshat-v1")
game = env.unwrapped

try:
    # Configure four players and initialise the experiment.
    game.startExperiment(
        gameType="POINTS",
        stopCriteria=1,
        playerNames=["P0", "P1", "P2", "P3"],
        logDirectory=r"D:\ChefsHat-RL-Task2\logs\smoke_test",
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
