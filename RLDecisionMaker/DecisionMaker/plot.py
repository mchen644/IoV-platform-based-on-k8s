import os
import matplotlib.pyplot as plt

def read_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(float(line.strip()))
    return data

def plot_line_chart(data):
    x = range(1, len(data) + 1)  # x轴为数据的行数
    plt.plot(x, data, marker='o')
    plt.xlabel('行数')
    plt.ylabel('数据')
    plt.title('数据趋势折线图')
    plt.grid(True)
    plt.savefig("line_chart.png")  # 保存折线图
    plt.clf()  # 清空图像
    print("折线图已保存为 line_chart.png")

file_path = "ep_reward.txt"
data = read_file(file_path)
plot_line_chart(data)

