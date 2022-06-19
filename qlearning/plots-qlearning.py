import pickle
import matplotlib.pyplot as plt

with open("results/qlearning-stats.dat", "rb") as fr:
    log = pickle.load(fr)

val = log[0]
time = log[1]

plt.plot(time, val)
plt.show()
