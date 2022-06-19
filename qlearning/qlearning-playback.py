import os
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from wrappers import wrapper
from network import *

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
frameSkipCount = 4
# Applies custom wrappers to the environment.
env = wrapper(env, shape=84, skip=frameSkipCount)

MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
env = JoypadSpace(env, MOVEMENT)


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