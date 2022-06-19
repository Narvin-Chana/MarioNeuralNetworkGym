import pickle

import gym_super_mario_bros
from deap import base, creator
from nes_py.wrappers import JoypadSpace

from wrappers import wrapper

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 3000

with open("bestind.hof", "rb") as fr:
    hof = pickle.load(fr)
    print(hof)

max_fit = 0
for ind in hof.items:
    if ind.fitness.values[0] > max_fit:
        best_ind = ind


CUSTOM_MOVEMENT = [["NOOP"], ["right", "A", "B"], ["right", "B"], ["left", "A", "B"], ["left", "B"]]

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, CUSTOM_MOVEMENT)
# Applies custom wrappers to the environment.
frameSkipCount = 4
env = wrapper(env, 84, frameSkipCount)


def evaluate_individual(ind):
    env.reset()

    current_step = 0
    total_reward = 0
    done = False

    while not done:
        if current_step >= IND_SIZE:
            break

        _, reward, d, info = env.step(ind[current_step])
        total_reward += reward
        done = d
        current_step += 1
        env.render()

    return total_reward,

evaluate_individual(best_ind)