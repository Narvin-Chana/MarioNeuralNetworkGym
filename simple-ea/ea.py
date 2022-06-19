import datetime

import collections.abc

# scoop needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

from scoop import futures
from deap import base, creator, tools, algorithms
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from qlearning.wrappers import wrapper

SAVE_INTERVAL = 1

NB_GEN = 400000
POP_SIZE = 50
IND_SIZE = 3000
NB_ACTIONS = 5

IND_CROSSOVER_RATE = 0.05
IND_MUTATION_RATE = 1
GENE_MUTATION_RATE = 0.01

SELECT_SIZE = 5

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


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

if __name__ == "__main__":
    import os
    import time
    import numpy
    from deap import creator
    from numpy import random
    import pickle

    RESULTS_DIR = "results/{0}/"
    result_name = datetime.datetime.now().strftime("%H-%M-%S")

    dir_name = RESULTS_DIR.format(result_name)
    path = dir_name + f"{result_name}." + "{}"
    os.makedirs(dir_name, exist_ok=True)


    def pickle_objects(objects):
        for (ext, o) in objects:
            with open(path.format(ext), "wb") as f:
                pickle.dump(o, f)


    start = time.time()

    toolbox = base.Toolbox()
    toolbox.register("attr_action", random.randint, 0, NB_ACTIONS)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_action, n=IND_SIZE)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=NB_ACTIONS - 1, indpb=GENE_MUTATION_RATE)
    toolbox.register("select", tools.selTournament, tournsize=SELECT_SIZE)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("map", futures.map)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    hof = tools.HallOfFame(10)
    history = tools.History()
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    pop = [toolbox.individual() for _ in range(POP_SIZE)]

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    history.update(pop)
    logbook.record(time=time.time() - start, gen=0, evals=len(invalid_ind), **stats.compile(pop))
    hof.update(pop)
    print(logbook.stream)

    for g in range(NB_GEN):
        # Select and clone the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        offspring = algorithms.varAnd(offspring, toolbox, IND_CROSSOVER_RATE, IND_MUTATION_RATE)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        logbook.record(time=time.time() - start, gen=g + 1, evals=len(invalid_ind), **stats.compile(pop))
        hof.update(pop)
        print(logbook.stream)

        if g % SAVE_INTERVAL == 0:
            pickle_objects([("log", logbook), ("hof", hof), ("hist", history)])
