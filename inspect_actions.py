
import gym
import numpy as np
import ChefsHatGym.env

env = gym.make("chefshat-v1")
game = env.unwrapped

try:
    game.startExperiment(
        gameType="POINTS",
        stopCriteria=1,
        playerNames=["P0", "P1", "P2", "P3"],
        logDirectory=r"D:\ChefsHat-RL-Task2\logs\step_test",
        verbose=0,
        saveLog=False,
        saveDataset=False,
    )

    observation, info = env.reset(seed=42)

    player = game.currentPlayer
    possible_actions = game.getPossibleActions(player)
    legal_indices = np.flatnonzero(possible_actions)

    print("Legal actions:", legal_indices.tolist())

    # Select the first legal action.
    chosen_index = int(legal_indices[0])

    # Create the 200-element action vector.
    action = np.zeros(200, dtype=np.float32)
    action[chosen_index] = 1.0

    observation, reward, terminated, truncated, info = env.step(action)

    print("Chosen action index:", chosen_index)
    print("Next observation shape:", observation.shape)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Action valid:", info.get("Action_Valid"))

finally:
    env.close()
