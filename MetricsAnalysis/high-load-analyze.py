import csv
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
        self.TaskNumber = []

    def process_data(self):
        self.CPULimit[-1] = 6000
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
    def get_mem(self, ratio=1000 * 1024 * 1024):
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
        count = 0
        for row in reader:
            cpuLimit = row["CPULimit"]
            metrics.CPULimit.append(int(cpuLimit[0:-1]) if cpuLimit != '0' else 0)
            metrics.AET.append(row["AET"])
            metrics.AvgCPUUsage.append(float(row["AvgCPUUsage"]))
            metrics.MaxCPUUsage.append(float(row["MaxCPUUsage"]))
            metrics.AvgMemoryUsage.append(float(row["AvgMemoryUsage"]))
            metrics.MaxMemoryUsage.append(float(row["MaxMemoryUsage"]))
            metrics.HighLoadRatio.append(float(row["HighLoadRatio"]))
            if "TaskNumber" in row:
                metrics.TaskNumber.append(int(row["TaskNumber"]))
            count = count + 1
            if count == 51:
                break
    return metrics


high_load_as1_metrics = read_metrics(R'D:\PyProject\ApplicationProfiler\data\High CPU Load Pods\1\slamMetrics_as1.csv')
high_load_controller_metrics = read_metrics(R'D:\PyProject\ApplicationProfiler\data\High CPU Load '
                                            R'Pods\1\slamMetrics_controller.csv')

single_pod_as1_metrics = read_metrics(R'D:\PyProject\ApplicationProfiler\data\Single Pod\4\slamMetrics_as1.csv')
single_pod_controller_metrics = read_metrics(
    R'D:\PyProject\ApplicationProfiler\data\Single Pod\4\slamMetrics_controller.csv')

high_load_controller_metrics_2 = read_metrics(R'D:\学术资料\硕士\CS 7638 AI Techniques for '
                                              R'Robotics\Project\RL_Introduction\MetricsAnalyse\data\High CPU Load '
                                              R'Pods\2\slamMetrics_controller.csv')
high_load_controller_metrics_2.process_data()

high_load_as1_metrics.process_data()
high_load_controller_metrics.process_data()
single_pod_as1_metrics.process_data()
single_pod_controller_metrics.process_data()


def plot_singlex(sub_plt, xlabel, ylabel, x, y, scatter=()):
    if len(scatter) == 0:
        scatter = [False] * len(y)
    i = 0
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for (data, label), sca in zip(y, scatter):
        if not sca:
            sub_plt.plot(x, data, label=label, color=colors[i])
        else:
            sub_plt.scatter(x, data, label=label, color=colors[i], s=8)
        i += 1
        i = i % len(colors)
    sub_plt.legend()
    sub_plt.set_xlabel(xlabel)
    sub_plt.set_ylabel(ylabel)
    uplimit = 0
    for data, _ in y:
        uplimit = max(uplimit, max(data))
    sub_plt.set_ylim(0, uplimit * 1.2)
    # plt.show()


def plot_multix(plt, xlabel, ylabel, data):
    for label, x, y in data:
        plt.plot(x, y, label=label)
    plt.legend()
    plt.set_xlabel(xlabel)
    plt.set_ylabel(ylabel)
    # plt.show()


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 6))

plot_singlex(axes[0, 0], "CPU Limit(mile-cores)", "AET(seconds)",
             high_load_controller_metrics.CPULimit,
             ((high_load_controller_metrics.AET, "controller-high-load"),
              (single_pod_controller_metrics.AET, "controller-single-pod"),
              (high_load_as1_metrics.AET, "AS1-high-load"),
              (single_pod_as1_metrics.AET, "AS1-single-pod")))
plot_singlex(axes[0, 1], "CPU Limit(mile-cores)", "Real Usage",
             high_load_controller_metrics.CPULimit,
             ((high_load_controller_metrics.AvgCPUUsage, "controller-avg"),
              (high_load_as1_metrics.AvgCPUUsage, "AS1-avg"),
              (high_load_controller_metrics.MaxCPUUsage, "controller-max"),
              (high_load_as1_metrics.MaxCPUUsage, "AS1-max"),
              (high_load_controller_metrics.CPULimit, "CPU limit")))

plot_multix(axes[0, 2], "CPU Limit Ratio", "AET(seconds)",
            (("controller-high-load", high_load_controller_metrics.get_cpu_ratio(32), high_load_controller_metrics.AET),
             ("controller-single-pod", single_pod_controller_metrics.get_cpu_ratio(32),
              single_pod_controller_metrics.AET),
             ("as1-high-load", high_load_as1_metrics.get_cpu_ratio(16), high_load_as1_metrics.AET),
             ("as1-single-pod", single_pod_as1_metrics.get_cpu_ratio(16), single_pod_as1_metrics.AET)))
plot_singlex(axes[1, 0], "CPU Limit(mile-cores)", "Max Task Number with High-Load",
             high_load_controller_metrics.CPULimit,
             ((high_load_controller_metrics.TaskNumber, "controller"), (high_load_as1_metrics.TaskNumber, "AS1")))

avg_mem_controller, max_mem_controller = high_load_controller_metrics.get_mem()
avg_mem_as1, max_mem_as1 = high_load_as1_metrics.get_mem()
plot_singlex(axes[1, 1], "CPU Limit(mile-cores)", "Mem Usage(MB)",
             high_load_controller_metrics.CPULimit,
             ((avg_mem_controller, "controller-avg"), (max_mem_controller, "controller-max"),
              (avg_mem_as1, "As1-avg"), (max_mem_as1, "As1-max")))

net = torch.nn.Sequential(
    torch.nn.Linear(in_features=1, out_features=10),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=10, out_features=10),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=10, out_features=1)
)

x = np.array(high_load_controller_metrics.CPULimit)

y = np.array(high_load_controller_metrics.AET)
x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32), dim=1)


# y = x + x.pow(2) + torch.rand(x.size())

def normalize(x):
    mean = torch.mean(x)
    std = torch.std(x)
    x_normalized = (x - mean) / std
    return x_normalized


x = normalize(x)
y = torch.tensor(y, dtype=torch.float32).reshape(len(y), 1)

loader = Data.DataLoader(
    dataset=Data.TensorDataset(x, y),
    batch_size=8,
    shuffle=True,
    num_workers=0
)
optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=0.001)
loss_func = torch.nn.MSELoss()

for epoch in range(500):
    for step, (batch_x, batch_y) in enumerate(loader):
        prediction = net(x)  # 喂给 net 训练数据 x, 输出预测值

        loss = loss_func(prediction, y)  # 计算两者的误差

        optimizer.zero_grad()  # 清空上一步的残余更新参数值

        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    print('Epoch: ', epoch, "Loss: ", loss.item())

predict_AET = net(normalize(torch.tensor(high_load_controller_metrics_2.CPULimit, dtype=torch.float32)
                  .reshape(len(high_load_controller_metrics_2.CPULimit), 1))).detach().numpy()

print("Loss is ", loss_func(
    torch.tensor(predict_AET, dtype=torch.float32).reshape(len(predict_AET), 1),
    torch.tensor(high_load_controller_metrics_2.AET, dtype=torch.float32).reshape(len(predict_AET), 1)).item())
print(torch.__version__)

plot_singlex(axes[1, 2], "CPU Limit(mile-cores)", "AET(seconds)",
             high_load_controller_metrics.CPULimit,
             ((high_load_controller_metrics_2.AET, "controller-high-load-test"),
              (predict_AET, "controller-predict")), (True, True))

diff_controller = np.array(high_load_controller_metrics.AET) / np.array(single_pod_controller_metrics.AET)
diff_as1 = np.array(high_load_as1_metrics.AET) / np.array(single_pod_as1_metrics.AET)

plot_multix(axes[2, 0], "TaskNumber", "AET Difference",
            (("controller", high_load_controller_metrics.TaskNumber, diff_controller),
             ("AS1", high_load_as1_metrics.TaskNumber, diff_as1)))

plt.show()
