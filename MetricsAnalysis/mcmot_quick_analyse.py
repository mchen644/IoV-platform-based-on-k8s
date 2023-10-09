import numpy as np
from matplotlib import pyplot as plt

frames_num = 647
task_numbers = np.array([1, 2, 3, 4, 5])
latencies = np.array([31.874, 50.3, 70, 87.7, 105.552])
total_latencies_per_task = np.divide(latencies, task_numbers)
avg_latency_per_frame = np.divide(latencies * 1000, frames_num)
total_latency = 5 * np.divide(latencies, task_numbers)

print(task_numbers, total_latencies_per_task, latencies, avg_latency_per_frame)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
axes[0, 0].plot(task_numbers, latencies, label="Latency(sec) Per Round")
axes[0, 1].plot(task_numbers, total_latencies_per_task, label="Avg Latency Per Task(sec)")
axes[1, 0].plot(task_numbers, avg_latency_per_frame, label="Avg Latency Per Frame(msec)")
axes[1, 1].plot(task_numbers, total_latency, label="Total Latency For 5 task(sec)")

for i in [0, 1]:
    for j in [0, 1]:
        axes[i, j].set_xlabel("Task Number at the Same Time")
        axes[i, j].set_xticks(task_numbers)
        axes[i, j].legend()

plt.xlabel("Task Number at the Same Time")
plt.xticks(task_numbers)
plt.legend()
plt.show()

