import os
import time

import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import Agent

import worldutils
import worldview
from wrappers import wrapper
from network import *

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, [["NOOP"], ["right", "A", "B"], ["right", "B"], ["left", "A", "B"], ["left", "B"]])
# Applies custom wrappers to the environment.
frameSkipCount = 4
env = wrapper(env, frameSkipCount)


def play_with_trained_model():
    file_dir = os.getcwd()
    network_filepath = os.path.join(file_dir, 'model.h5')
    # Could be used if we already have partially trained the network and save an intermediary best
    best_model = keras.models.load_model(network_filepath)

    done = True
    while True:
        if done:
            env.reset()
        state = worldutils.get_simplified_world(env)
        action_probs = best_model(np.array([state]))
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()

        _, reward, done, info = env.step(action)

        env.render()

    env.close()


def main():
    """
    An example pipeline of applying Q learning.
    Retrieved largely from: https://keras.io/examples/rl/deep_q_network_breakout/
    There are more techniques of course.
    Currently, network returns sampled numbers which still need to be translated to the actions
    known to the environment
    """
    file_dir = os.getcwd()
    network_filepath = os.path.join(file_dir, 'model.h5')
    n_actions = 5
    batch_size = 32  # Size of batch taken from replay buffer
    # Note: The Deepmind paper Mnih et al. (2013) suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    agent = Agent.QAgent(n_actions, lr=0.00025, gamma=0.99, epsilon=1.0, epsilon_min=0.1,
                         epsilon_max=1.0, batch_size=batch_size, max_mem_length=max_memory_length)

    max_steps_per_episode = 10000

    episode_reward_history = []
    running_reward = 0
    frame_count = 0

    # Number of frames to take random action and observe output
    max_episodes = 1000

    # Train the model after 4 actions
    update_after_actions = 4

    # How often to update the target network
    update_target_network = 10000

    t0 = time.time()

    for episode_count in range(max_episodes):
        print(f"Start of episode {episode_count}.")
        env.reset()
        episode_reward = 0

        state = worldutils.get_simplified_world(env)

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1

            action = agent.step(state, frame_count)

            # Apply the sampled action in our environment
            _, reward, done, info = env.step(action)

            env.render()
            # viewer.render(state)

            state_next = worldutils.get_simplified_world(env)

            episode_reward += reward

            # Save actions and states in replay buffer
            agent.remember(state, action, state_next, done, reward)

            state = state_next
            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(agent.memory) > batch_size:
                agent.update()

            if frame_count % update_target_network == 0:
                # update the target network with new weights
                agent.update_target_network()
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(agent.memory) > max_memory_length:
                agent.handle_mem()

            if done or info['flag_get']:
                print("Done!")
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        print(f"End of episode {episode_count}. Reward mean: {running_reward}")

        # if running_reward > 500:  # Condition to consider the task solved
        #     print("Solved at episode {}!".format(episode_count))
        #     break

    t1 = time.time()

    print("Time elapsed during execution: " + str(t1-t0))

    # Save best network

    agent.save_network(network_filepath)
    env.close()


# viewer = worldview.WorldViewer()

main()

# play_with_trained_model()

#
# done = True
# while True:
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     viewer.render(worldutils.get_simplified_world(env))
#     env.render()
#
# env.close()
