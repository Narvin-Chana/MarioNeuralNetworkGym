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

print(env.observation_space)


def play_with_trained_model():
    file_dir = os.getcwd()
    network_filepath = os.path.join(file_dir, 'models/06-10T00-19/EP1100')
    # Could be used if we already have partially trained the network and save an intermediary best
    best_model = keras.models.load_model(network_filepath)

    done = False
    state = env.reset()
    state = np.transpose(state, (1, 2, 0))
    total_reward = 0
    while True:
        env.render()
        if done:
            env.reset()

        action_probs = best_model(tf.expand_dims(state, 0))
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()

        state, reward, done, info = env.step(action)
        state = np.transpose(state, (1, 2, 0))
        total_reward += reward
        # print(f"Current total reward: {total_reward}")
    env.close()


def main():
    """
    An example pipeline of applying Q learning.
    Retrieved largely from: https://keras.io/examples/rl/deep_q_network_breakout/
    There are more techniques of course.
    Currently, network returns sampled numbers which still need to be translated to the actions
    known to the environment
    """
    time_values = []
    episode_rewards_full = []

    file_dir = os.getcwd()
    network_filepath = os.path.join(file_dir, 'models', datetime.datetime.now().strftime("%m-%dT%H-%M"))
    os.makedirs(network_filepath, exist_ok=True)
    os.makedirs(os.path.join(file_dir, 'results'), exist_ok=True)
    n_actions = len(MOVEMENT)
    batch_size = 32  # Size of batch taken from replay buffer

    # Note: The Deepmind paper Mnih et al. (2013) suggests 1000000 max_memory_length however this causes memory issues
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

    # Train the model after 4 actions
    update_after_actions = 4

    # Determines the checkpoint saving frequency
    save_after_episodes = 100

    # How often to update the target network
    update_target_network = 2500

    t0 = time.time()

    for episode_count in range(max_episodes):
        if time.time() - t0 > 10*60*60:
            break

        print(f"Start of episode {episode_count}.")
        state = env.reset()
        state = np.transpose(state, (1, 2, 0))
        episode_reward = 0
        nb_iter = 1

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
            # Update every fourth frame and once batch size is over 32
            if len(agent.memory) > batch_size:
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

            if done or info['flag_get']:
                state = env.reset()
                state = np.transpose(state, (1, 2, 0))
                nb_iter += 1

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward / nb_iter)
        episode_rewards_full.append(episode_reward / nb_iter)

        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        time_values.append(time.time() - t0)

        if episode_reward > best_fitness:
            best_fitness = episode_reward
        if episode_count % save_after_episodes == 0:
            agent.save_checkpoint(network_filepath, episode_count, episode_reward / nb_iter, running_reward, best_fitness, t0)

        print(f"End of episode {episode_count}. Episode reward: {episode_reward / nb_iter}. Reward total of episode: {episode_reward}. Num_iter: {nb_iter}. Reward mean: {running_reward}. Best "
              f"fitness: {best_fitness}")

        with open(os.path.join("results", "qlearning-stats.dat"), "wb") as fw:
            pickle.dump([episode_rewards_full, time_values], fw)


    t1 = time.time()

    print("Time elapsed during execution: " + str(t1 - t0))

    agent.save_network(network_filepath)
    env.close()


main()

# play_with_trained_model()
