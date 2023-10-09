from copy import deepcopy

from matplotlib import pyplot as plt
import csv
import numpy as np

dataset = 2

# 结构体定义
class Metrics:
    fusion_node_name: str
    gpu_limit: int
    cpu_limit: int
    max_cpu_usage: int
    avg_latency: float
    pass


# let error data in box plot also be considered inside the quarter
def adjust_dataset(data):
    # 找到每个数据集的上下四分位和异常值
    new_data = []
    outliers = [[] for _ in range(len(data))]

    for i, dataset in enumerate(data):
        Q1 = np.percentile(dataset, 25)
        Q3 = np.percentile(dataset, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 确定异常值
        outliers[i] = [value for value in dataset if value < lower_bound or value > upper_bound]
        # 结合原始数据和异常值
        combined_data = np.concatenate((dataset, outliers[i]))

        new_data.append(combined_data)
    return new_data


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


def getLatency(node_name, gpu_limit, cpu_limit, exclude_first=True):
    with open(
            'D:\学术资料\硕士\CS 7638 AI Techniques for Robotics\Project\RL_Introduction\MetricsAnalyse\data\CompleteTask'
            '\detail_metrics\\full_task\\{0}\latencies_{1}_{2}_{3}.csv'
                    .format(dataset, node_name, gpu_limit, cpu_limit),
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


cpu_limits = {
    "as1": list(range(400, 3200, 200)),
    "controller": list(range(1000, 5200, 200))
}
gpu_limits = [33, 50, 100]
total_medians = []
total_confidences = []
latencies = {
    "as1": [],
    "controller": []
}
latencies_map = {
    33: deepcopy(latencies),
    50: deepcopy(latencies),
    100: deepcopy(latencies)
}
for fusion_node in cpu_limits.keys():
    for gpu_limit in gpu_limits:
        for cpu_limit in cpu_limits[fusion_node]:
            latencies_map[gpu_limit][fusion_node].append(getLatency(fusion_node, gpu_limit, cpu_limit))
medians = {
    "as1": [],
    "controller": [],
}

medians_map = {
    33: deepcopy(medians),
    50: deepcopy(medians),
    100: deepcopy(medians)
}

for fusion_node in cpu_limits.keys():
    for gpu_limit in gpu_limits:
        for i, cpu_limit in enumerate(cpu_limits[fusion_node]):
            medians_map[gpu_limit][fusion_node].append(np.median(latencies_map[gpu_limit][fusion_node][i]))

confidence = {
    "as1": [],
    "controller": [],
}

confidence_map = {
    33: deepcopy(medians),
    50: deepcopy(medians),
    100: deepcopy(medians)
}

for fusion_node in cpu_limits.keys():
    for gpu_limit in gpu_limits:
        for i, cpu_limit in enumerate(cpu_limits[fusion_node]):
            confidence_map[gpu_limit][fusion_node].append(
                get_confidence_interval([latencies_map[gpu_limit][fusion_node][i]])[0])

for gpu_limit in gpu_limits:
    for fusion_node in cpu_limits.keys():
        print(cpu_limits[fusion_node])
        print(medians_map[gpu_limit][fusion_node])
        plt.plot(cpu_limits[fusion_node], medians_map[gpu_limit][fusion_node], label=fusion_node)
        plt.errorbar(cpu_limits[fusion_node], medians_map[gpu_limit][fusion_node],
                     yerr=confidence_map[gpu_limit][fusion_node], fmt='o', capsize=5, capthick=1,
                            ms=2, label=fusion_node)
    plt.title("GPU limit is " + str(gpu_limit))
    plt.legend()
    plt.show()
    plt.close()
