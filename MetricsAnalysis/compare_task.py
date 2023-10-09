import numpy as np
from matplotlib import pyplot as plt


def get_confidence_interval(data, confidence=0.95):
    from scipy.stats import sem, t
    means, confidence_intervals = [], []
    for dataset in data:
        mean = np.mean(dataset)
        std_err = sem(dataset)
        ci_range = std_err * t.ppf((1 + confidence) / 2, len(dataset) - 1)

        means.append(mean)
        confidence_intervals.append(ci_range)
    return confidence_intervals


def getLatency(measure_gpu, under_gpu, exclude_first=True):
    with open(
            'D:\学术资料\硕士\CS 7638 AI Techniques for Robotics\Project\RL_Introduction\MetricsAnalyse\data\CompleteTask'
            '\detail_metrics\compare_task\latency_{0}_under_{1}.csv'.format(measure_gpu, under_gpu),
            'r') as file:

        in_latencies = []
        if exclude_first:
            file.readline()
        while True:
            num = file.readline()
            if num == "":
                break
            in_latencies.append(float(num))
    return in_latencies


latency_25_50 = getLatency(25, 50)
latency_50_50 = getLatency(50, 50)
latencies = {}
medians = {}
means = {}
confidences = {}
for measure_gpu, under_gpu in [(25, 50), (50, 50)]:
    latencies[(measure_gpu, under_gpu)] = getLatency(measure_gpu, under_gpu)
    medians[(measure_gpu, under_gpu)] = np.median(latencies[(measure_gpu, under_gpu)])
    means[(measure_gpu, under_gpu)] = np.mean(latencies[(measure_gpu, under_gpu)])
    confidences[(measure_gpu, under_gpu)] = get_confidence_interval([latencies[(measure_gpu, under_gpu)]])[0]

plt.errorbar(["25% with 50%", "50 with 50%"], list(medians.values()), yerr=list(confidences.values()), fmt='o', capsize=5, capthick=1,
                        ecolor='black', color='blue', ms=2)
plt.show()
exit(0)

fig, axes = plt.subplots(1, 2)
fusion_node = "controller"
cpu_limits = list(range(1000, 5200, 200))
for i, gpu_limit in enumerate([16, 20, 25, 33, 50, 100]):
    x, y = i // 3, i % 3
    latencies = []
    for cpu_limit in cpu_limits:
        latencies.append(getLatency(fusion_node, gpu_limit, cpu_limit))

    medians = [np.median(latency) for latency in latencies]
    variances = [np.std(latency) for latency in latencies]
    means = [np.mean(latency) for latency in latencies]
    confidences = get_confidence_interval(latencies, confidence=0.99)
    axes[x][y].errorbar(cpu_limits, medians, yerr=confidences, fmt='o', capsize=5, capthick=1,
                        ecolor='black', color='blue', ms=2)
    axes[x][y].set_title("gpu limit {}% in {}".format(gpu_limit, fusion_node))
    # axes[x][y].set_ylim((40, 140))
    axes[x][y].set_ylim((0, 250))
    # axes[x][y].plot(cpu_limits, medians, label="median")
    axes[x][y].plot(cpu_limits, means, label="mean")
    axes[x][y].legend()

plt.show()
plt.close()
exit(0)