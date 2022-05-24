import time
from datetime import datetime

import numpy
from deap import base, creator, tools, algorithms
from numpy import random

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from wrappers import wrapper

NB_GEN = 10000
POP_SIZE = 25
IND_SIZE = 3000
NB_ACTIONS = 5

IND_CROSSOVER_RATE = 0.5
IND_MUTATION_RAT = 0.1
GENE_MUTATION_RATE = 1 / IND_SIZE

SELECT_SIZE = 2

CUSTOM_MOVEMENT = [["NOOP"], ["right", "A", "B"], ["right", "B"], ["left", "A", "B"], ["left", "B"]]

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, CUSTOM_MOVEMENT)
# Applies custom wrappers to the environment.
frameSkipCount = 4
env = wrapper(env, 4)


def evaluate_individual(ind):
    env.reset()

    current_step = 0
    total_reward = 0
    done = False

    while not done:
        _, reward, d, info = env.step(ind[current_step])
        total_reward += reward
        done = d
        current_step += 1
        env.render()

    return (total_reward,)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_action", random.randint, 0, NB_ACTIONS)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_action, n=IND_SIZE)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=NB_ACTIONS-1, indpb=GENE_MUTATION_RATE)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluate_individual)

pop = [toolbox.individual() for _ in range(POP_SIZE)]

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
hof = tools.HallOfFame(10)
finalpop, logbook = algorithms.eaSimple(pop, toolbox, IND_CROSSOVER_RATE, GENE_MUTATION_RATE, NB_GEN, stats=stats, halloffame=hof, verbose=True)

env.close()


with open("results/results_ea_{}.txt".format(datetime.now().strftime("%H-%M-%S")), "w") as fw:
    fw.write(str(hof))
    fw.write(str(logbook))