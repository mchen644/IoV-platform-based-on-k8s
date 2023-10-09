from copy import deepcopy

from matplotlib import pyplot as plt
import csv
import numpy as np
plt.style.use('ggplot')

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


def getLatency(dataset, task_name, node_name, resource_limit, path, exclude_first=True):
    with open(path.format(dataset, task_name, node_name, resource_limit),
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


def getLatency(dataset, node_name, gpu_limit, cpu_limit, path, exclude_first=True):
    with open(path.format(dataset, node_name, gpu_limit, cpu_limit),
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


def getIOLatency(dataset, path):
    with open(path.format(dataset)) as file:
        in_latencies = []
        while True:
            num = file.readline()
            if num == "":
                break
            in_latencies.append(float(num))
        return in_latencies


det_path = R'D:\学术资料\硕士\CS 7638 AI Techniques for Robotics\Project\RL_Introduction\MetricsAnalyse\data\DETTask' \
           R'\full_task\{0}\latencies_{1}_{2}_{3}.csv'
slam_path = R'D:\学术资料\硕士\CS 7638 AI Techniques for Robotics\Project\RL_Introduction\MetricsAnalyse\data\SLAMTask' \
            R'\3task\detail_metrics\{0}\latencies_{1}_{2}_{3}.csv'
complete_path = R'D:\学术资料\硕士\CS 7638 AI Techniques for ' \
                R'Robotics\Project\RL_Introduction\MetricsAnalyse\data\CompleteTask' \
                R'\detail_metrics\full_task\{0}\latencies_{1}_{2}_{3}.csv'
IO_path = R'D:\学术资料\硕士\CS 7638 AI Techniques for ' \
          R'Robotics\Project\RL_Introduction\MetricsAnalyse\data\ImageUpload' \
          R'\{0}\result.csv'

gpu_limits = [33, 50, 100]

cpu_limits = {
    "as1": list(range(400, 4200, 200)),
    "controller": list(range(1000, 5200, 200))
}

node_list = ["as1", "controller"]

slam_latencies = {
    "as1": [],
    "controller": []
}
slam_medians = deepcopy(slam_latencies)
slam_confidences = deepcopy(slam_latencies)

det_latencies, det_medians, det_confidences = [], [], []

complete_latencies = {
    "as1": {},
    "controller": {}
}
complete_medians = deepcopy(complete_latencies)

for node in node_list:
    for cpu_limit, gpu_limit in zip(cpu_limits[node], gpu_limits):
        latency = getLatency(2, node, gpu_limit, cpu_limit, complete_path)
        complete_latencies[node][(cpu_limit, gpu_limit)] = latency
        complete_medians[node][(cpu_limit, gpu_limit)] = np.median(latency)

for slam_node in node_list:
    for cpu_limit in cpu_limits[slam_node]:
        latency = getLatency(1, "slam", slam_node, cpu_limit, slam_path)
        slam_latencies[slam_node].append(latency)
        slam_medians[slam_node].append(np.median(latency))
        slam_confidences[slam_node].append(get_confidence_interval([latency])[0])

for gpu_limit in gpu_limits:
    latency = getLatency(1, "det", "gpu1", gpu_limit, det_path)
    det_latencies.append(latency)
    det_medians.append(np.median(latency))
    det_confidences.append(get_confidence_interval([latency])[0])

for fusion_node in cpu_limits.keys():
    plt.plot(cpu_limits[fusion_node], slam_medians[fusion_node], label="slam-" + fusion_node)
    plt.errorbar(cpu_limits[fusion_node], slam_medians[fusion_node],
                 yerr=slam_confidences[fusion_node], fmt='o', capsize=5, capthick=1,
                 ms=2)

for i, gpu_limit in enumerate(gpu_limits):
    x_range = list(range(400, 5200, 200))
    plt.plot(x_range, [det_medians[i]] * len(x_range), label='det-{0}'.format(gpu_limit))
    plt.errorbar(x_range, [det_medians[i]] * len(x_range),
                 yerr=[det_confidences[i]] * len(x_range), fmt='o', capsize=5, capthick=1,
                 ms=2)
plt.xlabel("CPU Limit(mcores)")
plt.ylabel("Latency(msec)")
plt.legend()
plt.show()
plt.close()


def draw_latency_component(resource_conf: list, det_latencies: list, slam_latencies: list,
                           total_latencies: list, IO_latencies: list):
    y1 = det_latencies
    y2 = slam_latencies

    bar_width = 0.3  # 条形宽度
    index_y1 = np.arange(len(resource_conf))  # y1条形图的横坐标
    index_y2 = index_y1 + bar_width  # y2条形图的横坐标
    index_y3 = (index_y2 + index_y1) / 2

    other_bottoms = []
    for i in range(len(y1)):
        other_bottoms.append(max(y1[i], y2[i]))
        total_latencies[i] -= other_bottoms[i]
        if total_latencies[i] <= 0:
            total_latencies[i] = 0

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_y1, height=y1, width=bar_width, label='det')
    plt.bar(index_y2, height=y2, width=bar_width, label='slam')

    plt.bar(index_y3, height=total_latencies, width=bar_width * 2, label='others', bottom=other_bottoms)
    plt.legend()  # 显示图例
    plt.xticks(index_y1 + bar_width / 2, resource_conf)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('latency(msec)')  # 纵坐标轴标题
    plt.title('Latency Component')  # 图形标题
    plt.ylim((0, 200))
    plt.show()
    plt.close()


IO_latencies = getIOLatency(1, IO_path)
image_count = {
    33: 5,
    50: 3,
    100: 1,
}

configs = [
    ['as1', 1200, 100], ['as1', 1200, 50], ['as1', 1200, 33], ['controller', 4000, 33],
    ['controller', 4000, 50],  ['controller', 4000, 100]
]
'''
configs = [
    ['controller', 3200, 100], ['controller', 3200, 50], ['as1', 1200, 33], ['controller', 3200, 33], ['controller', 4000, 50],
    ['as1', 2200, 100]
]
'''
dets, slams, totals, confs, io_latencies = [], [], [], [], []
for node, cpu_limit, gpu_limit in configs:
    dets.append(np.median(getLatency(1, "det", "gpu1", gpu_limit, det_path)))
    slams.append(np.median(getLatency(1, "slam", node, cpu_limit, slam_path)))
    totals.append(np.median(getLatency(2, node, gpu_limit, cpu_limit, complete_path)))
    io_latencies.append(IO_latencies[image_count[gpu_limit]])
    confs.append('({0},{1},{2})'.format(cpu_limit, gpu_limit, node))

print(dets, slams, totals, io_latencies)
draw_latency_component(confs, dets, slams, totals, io_latencies)