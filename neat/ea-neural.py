import multiprocessing
import pickle
import time

import worldutils
import os
import neat
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from wrappers import wrapper

SAVE_INTERVAL = 10

NB_GEN = 4
NB_ACTIONS = 7
CHECK_STEP = 60
X_POS_THRESHOLD = 50
STANDING_PENALTY = 500

CUSTOM_MOVEMENT = [["NOOP"], ["right", "A", "B"], ["right", "B"], ["left", "A", "B"], ["left", "B"]]

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# Applies custom wrappers to the environment.
frameSkipCount = 4
env = wrapper(env, 84, frameSkipCount)


class TimeReporter(neat.reporting.BaseReporter):
    def __init__(self, start):
        self.start = start
        self.time_values = []

    def post_evaluate(self, config, population, species, best_genome):
        self.time_values.append(time.time() - self.start)

    def get_time_values(self):
        return self.time_values


def evaluate_individual(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env.reset()

    current_step = 0
    total_reward = 0
    done = False

    while not done:

        state = worldutils.get_simplified_world(env)
        state = state.flatten()
        action = np.argmax(net.activate(state))
        _, reward, d, info = env.step(action)
        if current_step >= CHECK_STEP and info["x_pos"] < X_POS_THRESHOLD:
            break
        total_reward += reward
        done = d
        current_step += 1
        env.render()

    return total_reward


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    timer = TimeReporter(time.time())
    p.add_reporter(timer)

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_individual)
    winner = p.run(pe.evaluate, NB_GEN)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    return winner, stats, timer
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    winner, stats, timer = run(config_path)
    with open(os.path.join(local_dir, "../results", "neat", "neat.ind"), "w") as fw:
        fw.write(str(winner))
    with open(os.path.join(local_dir, "../results", "neat", "stats.dat"), "wb") as fw:
        pickle.dump([stats.get_fitness_stat(np.min), stats.get_fitness_stat(np.max), stats.get_fitness_stat(np.max), stats.get_fitness_stat(np.std), timer.get_time_values()], fw)
