import csv
from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np


class Metrics:
    def __init__(self):
        self.CPULimit = []
        self.AET = []
        self.AvgCPUUsage = []
        self.MaxCPUUsage = []
        self.AvgMemoryUsage = []
        self.MaxMemoryUsage = []
        self.HighLoadRatio = []

    def process_data(self):
        self.CPULimit[-1] = 8000
        for i in range(len(self.AET)):
            import re
            time_str = self.AET[i]
            pattern = r'^(\d+m)?(\d+(\.\d+)?s)?$'
            match = re.match(pattern, time_str)
            assert match
            minutes, seconds, _ = match.groups()
            minutes = int(minutes[:-1]) if minutes else 0
            seconds = float(seconds[:-1]) if seconds else 0
            self.AET[i] = minutes * 60 + seconds

            self.HighLoadRatio[i] = 1. / self.HighLoadRatio[i]

    def get_cpu_ratio(self, cores: int):
        ratio = deepcopy(self.CPULimit)
        for i in range(len(ratio)):
            ratio[i] = ratio[i] / (cores * 1000.)
        return ratio

    # default ratio=MB
    def get_mem(self, ratio=1000*1024*1024):
        avg_mem, max_mem = deepcopy(self.AvgMemoryUsage), deepcopy(self.MaxMemoryUsage)
        for i in range(len(avg_mem)):
            avg_mem[i] = avg_mem[i] / ratio
            max_mem[i] = max_mem[i] / ratio
        return avg_mem, max_mem


# 读取 CSV 文件
def read_metrics(road):
    with open(road, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        metrics = Metrics()
        for row in reader:
            cpuLimit = row["CPULimit"]
            metrics.CPULimit.append(int(cpuLimit[0:-1]) if cpuLimit != '0' else 0)
            metrics.AET.append(row["AET"])
            metrics.AvgCPUUsage.append(float(row["AvgCPUUsage"]))
            metrics.MaxCPUUsage.append(float(row["MaxCPUUsage"]))
            metrics.AvgMemoryUsage.append(float(row["AvgMemoryUsage"]))
            metrics.MaxMemoryUsage.append(float(row["MaxMemoryUsage"]))
            metrics.HighLoadRatio.append(float(row["HighLoadRatio"]))
    return metrics


# 打印结果
slam_as1_metrics = read_metrics(R'D:\PyProject\ApplicationProfiler\data\Single Pod\3\slamMetrics_as1.csv')
slam_controller_metrics = read_metrics(R'D:\PyProject\ApplicationProfiler\data\Single Pod\3\slamMetrics_controller.csv')
slam_as1_metrics.process_data()
slam_controller_metrics.process_data()
print(slam_as1_metrics.CPULimit)
print(slam_controller_metrics.AET)


def plot_singlex(xlabel, ylabel, x, y):
    for data, label in y:
        plt.plot(x, data, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    uplimit = 0
    for data, _ in y:
        uplimit = max(uplimit, max(data))
    plt.ylim(0, uplimit * 1.2)
    plt.show()


def plot_multix(xlabel, ylabel, data):
    for label, x, y in data:
        plt.plot(x, y, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


plot_singlex("CPU Limit(mile-cores)", "Application Execution Time(seconds)", slam_controller_metrics.CPULimit,
             ((slam_controller_metrics.AET, "controller"), (slam_as1_metrics.AET, "AS1")))
plot_singlex("CPU Limit(mile-cores)", "Real Usage",
             slam_controller_metrics.CPULimit,
             ((slam_controller_metrics.AvgCPUUsage, "controller-avg"), (slam_as1_metrics.AvgCPUUsage, "AS1-avg"),
              (slam_controller_metrics.MaxCPUUsage, "controller-max"), (slam_as1_metrics.MaxCPUUsage, "AS1-max"),
              (slam_controller_metrics.CPULimit, "CPU limit")))

plot_multix("CPU Limit Ratio", "Application Execution Time(seconds)",
            (("controller", slam_controller_metrics.get_cpu_ratio(32), slam_controller_metrics.AET),
             ("as1", slam_as1_metrics.get_cpu_ratio(16), slam_as1_metrics.AET)))

avg_mem_controller, max_mem_controller = slam_controller_metrics.get_mem()
avg_mem_as1, max_mem_as1 = slam_as1_metrics.get_mem()
plot_singlex("CPU Limit(mile-cores)", "Mem Usage(MB)",
             slam_controller_metrics.CPULimit,
             ((avg_mem_controller, "controller-avg"), (max_mem_controller, "controller-max"),
              (avg_mem_as1, "As1-avg"), (max_mem_as1, "As1-max")))

