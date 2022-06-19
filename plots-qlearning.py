import pickle
import matplotlib.pyplot as plt

with open("results/qlearning-stats(1).dat", "rb") as fr:
    log = pickle.load(fr)

time = log[0]
val = log[1]

plt.plot(time, val)
plt.show()