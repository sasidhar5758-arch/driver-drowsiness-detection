import matplotlib.pyplot as plt

fatigue_levels = [
    0,0,0,1,1,1,1,2,2,2
]

time = list(range(len(fatigue_levels)))

plt.plot(time, fatigue_levels)

plt.xlabel("Time Interval")

plt.ylabel("Fatigue Level")

plt.title("Driver Fatigue Progression")

plt.show()