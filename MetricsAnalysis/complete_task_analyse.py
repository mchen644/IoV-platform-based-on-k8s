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


fig, axes = plt.subplots(2, 2)

fusion_node = "controller"
if fusion_node == "as1":
    cpu_limits = list(range(400, 3200, 200))
elif fusion_node == "controller":
    cpu_limits = list(range(1000, 5200, 200))
gpu_limits = [33, 50, 100]
total_medians = []
total_confidences = []
for i, gpu_limit in enumerate(gpu_limits):
    x, y = i // 2, i % 2
    latencies = []
    for cpu_limit in cpu_limits:
        latencies.append(getLatency(fusion_node, gpu_limit, cpu_limit))

    medians = [np.median(latency) for latency in latencies]
    total_medians.append(medians)
    variances = [np.std(latency) for latency in latencies]
    means = [np.mean(latency) for latency in latencies]
    print("gpu {}, medians is {}".format(gpu_limit, medians))
    confidences = get_confidence_interval(latencies, confidence=0.95)
    total_confidences.append(confidences)
    axes[x][y].errorbar(cpu_limits, medians, yerr=confidences, fmt='o', capsize=5, capthick=1,
                        ecolor='black', color='blue', ms=2)
    axes[x][y].set_title("gpu limit {}% in {}".format(gpu_limit, fusion_node))
    axes[x][y].set_ylim((40, 400))
    # axes[x][y].set_ylim((0, 250))
    # axes[x][y].plot(cpu_limits, medians, label="median")
    #axes[x][y].plot(cpu_limits, means, label="mean")
    #axes[x][y].legend()

plt.show()
plt.close()

for gpu_limit, medians, confidences in zip(gpu_limits, total_medians, total_confidences):
    plt.plot(cpu_limits, medians, label='gpu limit {}'.format(gpu_limit))
    plt.errorbar(cpu_limits, medians, yerr=confidences, fmt='o', capsize=5, capthick=1,
                        ms=2, label="gpu limit {}".format(gpu_limit))
plt.legend()
plt.show()
plt.close()

for i, gpu_limit in enumerate(gpu_limits):
    x, y = i // 3, i % 3
    latencies = []
    for cpu_limit in cpu_limits:
        latencies.append(getLatency(fusion_node, gpu_limit, cpu_limit))

    medians = [np.median(latency) for latency in latencies]
    variances = [np.std(latency) for latency in latencies]
    means = [np.mean(latency) for latency in latencies]
    confidences = get_confidence_interval(latencies, confidence=0.99)
    plt.boxplot(latencies, labels=cpu_limits,
                       meanprops={'marker': 'v', 'color': 'green'},
                       showmeans=True,
                       showfliers=False)
    plt.title("gpu limit {}% in {}".format(gpu_limit, fusion_node))
    plt.show()

exit(0)

# 读取CSV文件
with open(
        'D:\学术资料\硕士\CS 7638 AI Techniques for Robotics\Project\RL_Introduction\MetricsAnalyse\data\CompleteTask\without_gpu.csv',
        'r') as file:
    reader = csv.DictReader(file)
    metrics = []
    for row in reader:
        metric = Metrics()
        for key, value in row.items():
            setattr(metric, key.strip(), value.strip())
        metrics.append(metric)

    x = [int(metric.gpu_limit) for metric in metrics]
    y = [float(metric.avg_latency) for metric in metrics]
    plt.xticks(x)
    plt.plot(x, y, "x-")
    plt.show()
