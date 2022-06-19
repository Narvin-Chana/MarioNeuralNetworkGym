import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("../results/13-38-06/13-38-06.log", "rb") as fr:
    log = pickle.load(fr)

time = [value["time"] for value in log]
max = [value["max"] for value in log]

plt.plot(time, max)
plt.title("Best fitness in population of size 50 over time (CEA)")
plt.xlabel("Time (s)")
plt.ylabel("Fitness score")
plt.show()

max_alltime = [np.max(max[0:i+1]) for i in range(len(log))]

plt.plot(time, max_alltime)
plt.title("Best fitness found over time (CEA)")
plt.xlabel("Time (s)")
plt.ylabel("Fitness score")
plt.show()

