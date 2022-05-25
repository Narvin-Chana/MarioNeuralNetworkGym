from scoop import futures
from deap import base, creator, tools, algorithms
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from wrappers import wrapper

NB_GEN = 1000
POP_SIZE = 100
IND_SIZE = 3000
NB_ACTIONS = 5

IND_CROSSOVER_RATE = 0.5
IND_MUTATION_RATE = 0.3
GENE_MUTATION_RATE = 100 / IND_SIZE

SELECT_SIZE = 10

CUSTOM_MOVEMENT = [["NOOP"], ["right", "A", "B"], ["right", "B"], ["left", "A", "B"], ["left", "B"]]

if __name__ != "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    # Applies custom wrappers to the environment.
    frameSkipCount = 4
    env = wrapper(env, frameSkipCount)


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


    return (total_reward,)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

if __name__ == "__main__":
    import numpy
    from deap import creator
    from numpy import random

    toolbox = base.Toolbox()
    toolbox.register("attr_action", random.randint, 0, NB_ACTIONS)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_action, n=IND_SIZE)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=NB_ACTIONS - 1, indpb=GENE_MUTATION_RATE)
    toolbox.register("select", tools.selRoulette)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("map", futures.map)

    pop = [toolbox.individual() for _ in range(POP_SIZE)]

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    hof = tools.HallOfFame(10)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

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
        print(stats.compile(pop))


# with open("results/results_ea_{}.txt".format(datetime.now().strftime("%H-%M-%S")), "w") as fw:
#    fw.write(str(hof))
#    fw.write(str(logbook))