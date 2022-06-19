import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("../results/neat/stats.dat", "rb") as fr:
    log = pickle.load(fr)

time = log[-1]
max = log[1]

plt.plot(time, max)
plt.title("Best fitness in population of size 20 over time (NEAT)")
plt.xlabel("Time (s)")
plt.ylabel("Fitness score")
plt.show()