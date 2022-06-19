import datetime
import os
import pickle
import time

import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace

import Agent
from network import *
from wrappers import wrapper

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
frameSkipCount = 4
# Applies custom wrappers to the environment.
env = wrapper(env, shape=84, skip=frameSkipCount)

MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
env = JoypadSpace(env, MOVEMENT)


def main():
    """
    Main loop for Q-learning
    """
    time_values = []
    episode_rewards_full = []

    file_dir = os.getcwd()
    network_filepath = os.path.join(file_dir, 'models', datetime.datetime.now().strftime("%m-%dT%H-%M"))
    os.makedirs(network_filepath)
    os.makedirs(os.path.join(file_dir, 'results'), exist_ok=True)
    n_actions = len(MOVEMENT)
    batch_size = 32  # Size of batch taken from replay buffer

    # max_memory_length to avoid memory issues
    max_memory_length = 50000
    agent = Agent.QAgent(n_actions, lr=0.00025, gamma=0.95, epsilon=1, epsilon_decay=0.99, epsilon_min=0.01,
                         epsilon_max=1.0, batch_size=batch_size, max_mem_length=max_memory_length)

    max_steps_per_episode = 10000

    episode_reward_history = []
    running_reward = 0
    best_fitness = 0
    frame_count = 0

    # Number of frames to take random action and observe output
    max_episodes = 1000000

    # Determines the checkpoint saving frequency
    save_after_episodes = 100

    # How often to update the target network
    update_target_network = 1e4

    t0 = time.time()

    for episode_count in range(max_episodes):
        if time.time() - t0 > 20 * 60 * 60:
            break

        print(f"Start of episode {episode_count}.")
        state = env.reset()
        state = np.transpose(state, (1, 2, 0))
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1

            env.render()

            action = agent.step(state, frame_count)

            # Apply the sampled action in our environment
            state_next, reward, done, info = env.step(action)
            state_next = np.transpose(state_next, (1, 2, 0))

            episode_reward += reward

            # Save actions and states in replay buffer
            agent.remember(state, action, state_next, done, reward)

            state = state_next
            # Update once batch size is over 32
            if len(agent.memory) >= batch_size:
                agent.update()

            if frame_count % update_target_network == 0:
                # update the target network with new weights
                agent.update_target_network()
                # Log details
                template = "Updated target network at episode {}, frame count {}"
                print(template.format(episode_count, frame_count))

            # Limit the state and reward history
            if len(agent.memory) > max_memory_length:
                agent.handle_mem()

            if done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        episode_rewards_full.append(episode_reward)

        if len(episode_reward_history) > 100:
            del episode_reward_history[0]
        running_reward = np.mean(episode_reward_history)

        time_values.append(time.time() - t0)

        if episode_reward > best_fitness:
            best_fitness = episode_reward
            # agent.save_checkpoint(network_filepath, episode_count, episode_reward, running_reward, best_fitness, t0,
            #                       is_best=True)
        if episode_count % save_after_episodes == 0:
            agent.save_checkpoint(network_filepath, episode_count, episode_reward, running_reward, best_fitness, t0)

        print(f"End of episode {episode_count}. Episode reward: {episode_reward}. Reward mean: {running_reward}. Best "
              f"fitness: {best_fitness}")

        with open(os.path.join("../results", "qlearning-stats.dat"), "wb") as fw:
            pickle.dump([episode_rewards_full, time_values], fw)

    t1 = time.time()

    print("Time elapsed during execution: " + str(t1 - t0))

    agent.save_network(network_filepath)
    env.close()


main()