from copy import deepcopy

from matplotlib import pyplot as plt
import csv
import numpy as np

plt.style.use('seaborn-paper')


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


def group_latencies(latencies: dict, configs: list, method='mean', confidence=95):
    """
    :param latencies latencies dict
    :param configs config list, each config must be [fusion_node, cpu_limit_gpu_limit]
    :param method means or medians to stat latencies
    :param confidence confidence interval
    :returns dets, slams, io_det, io_slams, totals, fusion, confs
    """

    def append(latency: list, confidence):
        return latency

    func = get_confidence_means if method == 'mean' else \
        get_confidence_medians if method == 'median' else \
            append if method == 'append' else \
                None
    assert func is not None
    dets, slams, io_det, io_slams, totals, fusion, confs = [], [], [], [], [], [], []
    for node, cpu_limit, gpu_limit in configs:
        dets.append(func(latencies[node][gpu_limit][cpu_limit]['det_proc'], confidence))
        slams.append(func(latencies[node][gpu_limit][cpu_limit]['slam_proc'], confidence))
        io_det.append(func(latencies[node][gpu_limit][cpu_limit]['det_io'], confidence))
        io_slams.append(func(latencies[node][gpu_limit][cpu_limit]['slam_io'], confidence))
        totals.append(func(latencies[node][gpu_limit][cpu_limit]['total'], confidence))
        fusion.append(func(latencies[node][gpu_limit][cpu_limit]['fusion'], confidence))
        confs.append('({0:.2f}%,{1}%)'.format(100 * cpu_limit / 16000, gpu_limit))
    return dets, slams, io_det, io_slams, totals, fusion, confs


def get_confidence_means(data, confidence=95):
    data = np.array(data)
    lower_bound = np.percentile(data, (100 - confidence) / 2)
    upper_bound = np.percentile(data, 100 - (100 - confidence) / 2)
    interval_data = []
    for latency in data:
        if lower_bound <= latency <= upper_bound:
            interval_data.append(latency)
    return np.mean(interval_data)


def get_confidence_medians(data, confidence=80):
    data = np.array(data)
    lower_bound = np.percentile(data, (100 - confidence) / 2)
    upper_bound = np.percentile(data, 100 - (100 - confidence) / 2)
    interval_data = []
    for latency in data:
        if lower_bound <= latency <= upper_bound:
            interval_data.append(latency)
    return np.median(interval_data)


def get_confidence_medians_index(data, confidence=80):
    data = np.array(data)
    lower_bound = np.percentile(data, (100 - confidence) / 2)
    upper_bound = np.percentile(data, 100 - (100 - confidence) / 2)
    interval_data = []
    for latency in data:
        if lower_bound <= latency <= upper_bound:
            interval_data.append(latency)
    m = np.median(interval_data)
    for i, num in enumerate(interval_data):
        if num == m:
            return i


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


def getLatency(dataset, node_name, gpu_limit, cpu_limit, path, exclude_first=True):
    with open(path.format(dataset, node_name, gpu_limit, cpu_limit),
              'r') as file:

        in_latencies = {
            'total': [],
            'slam_io': [],
            'slam_proc': [],
            'det_io': [],
            'det_proc': [],
            'fusion': [],
        }
        if exclude_first:
            file.readline()
        while True:
            num = file.readline()
            if num == "":
                break
            nums = num.split(' ')

            in_latencies['total'].append(float(nums[0]))
            in_latencies['slam_io'].append(float(nums[1]))
            in_latencies['slam_proc'].append(float(nums[2]))
            in_latencies['det_io'].append(float(nums[3]))
            in_latencies['det_proc'].append(float(nums[4]))
            in_latencies['fusion'].append(float(nums[5]) / 1000)
    return in_latencies


aggregated_path = R'D:\学术资料\硕士\CS 7638 AI Techniques for Robotics\Project\RL_Introduction\MetricsAnalyse\data' \
                  R'\AggregatedComponent\detail_metrics\{0}\latencies_{1}_{2}_{3}.csv'

gpu_limits = [33, 50, 100]

gpu_latencies = {
    33: {},
    50: {},
    100: {},
}
# latencies[fusion_node][gpu_limit][cpu_limit][latency_type]
latencies = {
    "as1": deepcopy(gpu_latencies),
    "controller": deepcopy(gpu_latencies),
}

cpu_limits = {
    "as1": list(range(400, 3100, 100)),
    "controller": list(range(1000, 6200, 200))
}

node_list = ['controller', "as1"]
dataset = 5

for fusion_node in node_list:
    for gpu_limit in gpu_limits:
        for cpu_limit in cpu_limits[fusion_node]:
            latency = getLatency(dataset, fusion_node, gpu_limit, cpu_limit, aggregated_path)
            latencies[fusion_node][gpu_limit][cpu_limit] = latency

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
        latency = latencies[node][gpu_limit][cpu_limit]['total']
        complete_latencies[node][(cpu_limit, gpu_limit)] = latency
        complete_medians[node][(cpu_limit, gpu_limit)] = get_confidence_medians(latency)

for slam_node in node_list:
    for cpu_limit in cpu_limits[slam_node]:
        latency = latencies[slam_node][100][cpu_limit]['slam_proc']
        slam_latencies[slam_node].append(latency)
        slam_medians[slam_node].append(get_confidence_means(latency))
        slam_confidences[slam_node].append(get_confidence_interval([latency])[0])

for gpu_limit in gpu_limits:
    latency = latencies['as1'][gpu_limit][3000]['det_proc']
    det_latencies.append(latency)
    det_medians.append(get_confidence_means(latency))
    det_confidences.append(get_confidence_interval([latency])[0])


def draw_IO_over_cpu(cpu_limits, slam_io, gpu_limit):
    plt.plot(cpu_limits, [get_confidence_means(io, 95) for io in slam_io], label='slam_io', color='b')
    # plt.errorbar(cpu_limits, [get_confidence_means(io, 90) for io in slam_io],
    # get_confidence_interval(slam_io, 0.90), color='b')
    plt.scatter(cpu_limits, [get_confidence_means(io, 95) for io in slam_io], marker='*')
    plt.title("IO Latency over CPU Resources with {0} task".format(int(100 / gpu_limit)))
    plt.xlabel("CPU Resources (mcores)")
    plt.ylabel("Latency (msec)")
    plt.ylim((0, 100))
    plt.legend()
    plt.show()
    plt.close()


io_cpu_node_name = 'as1'
io_cpu_cpu_limits = list(range(600, 2800, 100))
_, _, _, io_slams, _, _, _ = \
    group_latencies(latencies,
                    configs=[[io_cpu_node_name, cpu_limit, 33] for cpu_limit in io_cpu_cpu_limits],
                    method='append', confidence=100)
draw_IO_over_cpu(io_cpu_cpu_limits, io_slams, 33)


def draw_io_over_task_number():
    node_name = 'as1'
    cpu_limit = 1000
    _, _, io_dets, io_slams, _, _, _ = \
        group_latencies(latencies,
                        configs=[[node_name, cpu_limit, gpu_limit] for gpu_limit in gpu_limits],
                        method='mean', confidence=95)
    bar_width = 0.3  # 条形宽度
    index_y1 = np.arange(len(gpu_limits))  # y1条形图的横坐标
    index_y2 = index_y1 + bar_width  # y2条形图的横坐标
    plt.bar(index_y1, io_slams, color='b', width=bar_width, label='slam_io')
    plt.bar(index_y2, io_dets, color='g', width=bar_width, label='det_io')
    plt.title("IO Latency Over Task Number with cpu limit {}".format(cpu_limit))
    plt.xticks(index_y1 + bar_width / 2, [100 // gpu_limit for gpu_limit in gpu_limits])
    plt.xlabel("Task Number")
    plt.ylabel("Latency (msec)")
    plt.legend()
    plt.show()
    plt.close()


draw_io_over_task_number()


def draw_cpu_gpu_intersection():
    cores = {
        'as1': 16000,
        'controller': 32000
    }
    for fusion_node in node_list:
        _, slams_mean, _, _, _, _, _ = group_latencies(
            latencies,
            configs=[[fusion_node, cpu_limit, 50] for cpu_limit in cpu_limits[fusion_node]]
        )
        _, slams, _, _, _, _, _ = group_latencies(
            latencies,
            configs=[[fusion_node, cpu_limit, 50] for cpu_limit in cpu_limits[fusion_node]],
            method='append'
        )
        cpu_percentage = [100*cpu_limit/cores[fusion_node] for cpu_limit in cpu_limits[fusion_node]]
        plt.plot(cpu_percentage, slams_mean, label="slam-" + fusion_node)
        # plt.plot(cpu_limits[fusion_node], slam_medians[fusion_node], label="slam-" + fusion_node)
        plt.errorbar(cpu_percentage, slams_mean,
                     yerr=get_confidence_interval(slams), fmt='o', capsize=5, capthick=1,
                     ms=2)

    for i, gpu_limit in enumerate(gpu_limits):
        x_range = list(range(1, 21, 1))
        plt.plot(x_range, [det_medians[i]] * len(x_range), label='det-{0}'.format(gpu_limit))
        plt.errorbar(x_range, [det_medians[i]] * len(x_range),
                     yerr=[det_confidences[i]] * len(x_range), fmt='o', capsize=5, capthick=1,
                     ms=2)
    plt.xlabel("CPU Percentage")
    plt.ylabel("Latency(msec)")
    plt.xticks(list(range(0, 21, 1)))
    plt.ylim((0, 250))
    plt.title("CPU&GPU Latency Intersection")
    plt.legend()
    plt.show()
    plt.close()


draw_cpu_gpu_intersection()


def draw_latency_component(resource_conf: list, det_latencies: list, slam_latencies: list,
                           total_latencies: list, IO_latencies: list):
    y1 = det_latencies
    y2 = slam_latencies

    bar_width = 0.3  # 条形宽度
    index_y1 = np.arange(len(resource_conf))  # y1条形图的横坐标
    index_y2 = index_y1 + bar_width  # y2条形图的横坐标
    index_y3 = (index_y2 + index_y1) / 2

    other_bottoms = []
    io_bottoms = []
    for i in range(len(y1)):
        io_bottoms.append(max(y1[i], y2[i]))
        other_bottoms.append(IO_latencies[i])
        IO_latencies[i] -= io_bottoms[i]
        total_latencies[i] -= other_bottoms[i]
        if total_latencies[i] <= 0:
            assert False
            total_latencies[i] = 0
    print(io_bottoms, IO_latencies, total_latencies)

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_y1, height=y1, width=bar_width, label='det')
    plt.bar(index_y2, height=y2, width=bar_width, label='slam')

    plt.bar(index_y3, height=IO_latencies, width=bar_width * 2, label='IO', bottom=io_bottoms)
    plt.bar(index_y3, height=total_latencies, width=bar_width * 2, label='others', bottom=other_bottoms)
    plt.legend()  # 显示图例
    plt.xticks(index_y1 + bar_width / 2, resource_conf)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('latency(msec)')  # 纵坐标轴标题
    plt.title('Latency Component')  # 图形标题
    plt.show()
    plt.close()


def draw_latency_component_separate(resource_conf: list, det_latencies: list, slam_latencies: list,
                                    total_latencies: list, det_io_latencies: list, slam_io_latencies,
                                    fusion_latencies: list):
    bar_width = 0.3  # 条形宽度
    index_y1 = np.arange(len(resource_conf)) + 0.2  # y1条形图的横坐标
    index_y2 = index_y1 + bar_width  # y2条形图的横坐标
    index_y3 = (index_y2 + index_y1) / 2

    for i in range(len(det_latencies)):
        if i % 2 != 0:
            index_y1[i] -= .3
            index_y2[i] -= .3
            index_y3[i] -= .3

    fusion_bottom = []
    total_latencies = []
    for i in range(len(det_latencies)):
        fusion_bottom.append(max(
            det_latencies[i] + det_io_latencies[i],
            slam_latencies[i] + slam_io_latencies[i]
        ))
        total_latencies.append(fusion_bottom[i] + fusion_latencies[i])

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_y1, height=det_latencies, width=bar_width, label='det')
    plt.bar(index_y2, height=slam_latencies, width=bar_width, label='slam')
    plt.bar(index_y1, height=det_io_latencies, width=bar_width, label='det_io', bottom=det_latencies)
    plt.bar(index_y2, height=slam_io_latencies, width=bar_width, label='slam_io', bottom=slam_latencies)

    plt.bar(index_y3, height=fusion_latencies, width=bar_width * 2, label='fusion', bottom=fusion_bottom)
    plt.legend()  # 显示图例
    for i in range(len(det_latencies)):
        plt.text((np.add(index_y1, index_y2) / 2)[i], np.add(total_latencies, 5)[i],
                 'Total {0:.2f} msec'.format(total_latencies[i]), fontsize=8, ha='center',
                 va='bottom')
    plt.xticks(index_y1 + bar_width / 2, resource_conf)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('Latency (msec)')  # 纵坐标轴标题
    plt.ylim(0, np.max(total_latencies) + 30)
    plt.xlabel('(CPU Limit%, GPU Limit%)')
    plt.title('Latency Component (msec)')  # 图形标题
    plt.show()
    plt.close()


image_count = {
    33: 5,
    50: 3,
    100: 1,
}


def draw_configs(configs):
    dets, slams, totals, io_det, io_slams, confs, = [], [], [], [], [], []
    for node, cpu_limit, gpu_limit in configs:
        dets.append(np.median(latencies[node][gpu_limit][cpu_limit]['det_proc']))
        slams.append(np.median(latencies[node][gpu_limit][cpu_limit]['slam_proc']))
        io_det.append(np.median(latencies[node][gpu_limit][cpu_limit]['det_io']))
        io_slams.append(np.median(latencies[node][gpu_limit][cpu_limit]['slam_io']))
        totals.append(np.median(latencies[node][gpu_limit][cpu_limit]['total']))
        confs.append('({0},{1},{2})'.format(cpu_limit, gpu_limit, node))

    io_latencies = []
    for det_io, det_proc, slam_io, slam_proc in zip(io_det, dets, io_slams, slams):
        if det_io + det_proc > slam_io + slam_proc:
            io_latencies.append(det_io + det_proc)
        else:
            io_latencies.append(slam_io + slam_proc)

    draw_latency_component(confs, dets, slams, totals, io_latencies)


def draw_configs_seperate(configs):
    _, slams, _, io_slams, totals, fusions, confs = group_latencies(latencies, configs)
    gpu_configs = []
    for config in configs:
        gpu_configs.append(['as1', 3000, config[2]])
    dets, _, io_det, _, _, _, _ = group_latencies(latencies, gpu_configs)
    draw_latency_component_separate(confs, dets, slams, totals, io_det, io_slams, fusions)


configs = [
    ['as1', 1000, 33], ['as1', 2000, 33],
    ['as1', 1200, 50], ['as1', 2000, 50],
    ['as1', 1900, 100], ['as1', 400, 100]
]
draw_configs_seperate(configs)


def draw_fusion_low():
    configs = [['as1', 1000, 33]]
    _, slams, _, io_slams, _, fusions, confs = group_latencies(latencies, configs, method='append')
    gpu_configs = []
    for config in configs:
        gpu_configs.append(['as1', 3000, config[2]])
    dets, _, io_det, _, _, _, _ = group_latencies(latencies, gpu_configs, method='append')
    slams = slams[0]
    io_slams = io_slams[0]
    fusions = fusions[0]
    dets = dets[0]
    io_det = io_det[0]

    p95 = {
        'slams': np.percentile(slams, 95),
        'io_slams': np.percentile(io_slams, 95),
        'det': np.percentile(dets, 95),
        'io_dets': np.percentile(io_det, 95),
        'fusion': np.percentile(fusions, 95)
    }

    best = {
        'slams': np.min(slams),
        'io_slams': np.min(io_slams),
        'det': np.min(dets),
        'io_dets': np.min(io_det),
        'fusion': np.min(fusions)
    }

    mean = {
        'slams': get_confidence_means(slams),
        'io_slams': get_confidence_means(io_slams),
        'det': get_confidence_means(dets),
        'io_dets': get_confidence_means(io_det),
        'fusion': get_confidence_means(fusions)
    }

    # 设置条形图的位置和宽度
    bar_width = 0.4

    for base, data, name in zip([1, 2, 3], [p95, mean, best], ['95%', 'best', 'mean']):
        # 分别绘制slams与io_slams、det与io_dets两组并行的横向条形图
        perception = max(data['slams'] + data['io_slams'], data['det'], data['io_dets'])
        plt.barh([base], perception, height=bar_width, color='salmon', edgecolor='black', hatch='//')
        plt.barh([base], data['fusion'], height=bar_width, left=perception, color='palegreen', edgecolor='black', hatch='||')

    plt.yticks([1, 2, 3], ['95-percentile', 'Mean', 'Best-Case'])

    plt.xlabel('Latency (msec)')
    plt.title('Latency Distribution')

    import matplotlib.patches as mpatches
    patch1 = mpatches.Patch(facecolor='salmon', label='Perception', hatch='//')
    patch2 = mpatches.Patch(facecolor='palegreen', label='Fusion', hatch='||')

    plt.legend(handles=[patch1, patch2])
    plt.show()
    plt.close()


draw_fusion_low()
