import numpy as np
import matplotlib.pyplot as plt
import time
# 设置参数
arrival_rate = 1.0           # 平均到达率，即单位时间内预期到达的车辆数量

# 生成时间序列
times = np.arange(0, 3, 40/3600)  # 假设时间范围是0到24小时，以0.1小时(6 min, 360 s)为间隔

# 生成泊松分布随机变量
arrivals = np.random.poisson(arrival_rate, len(times))
print(arrivals)
print("haha")
# 绘制图形
plt.plot(times, arrivals)
plt.xlabel('Time (hours)')
plt.ylabel('Number of Arrivals')
plt.title('Vehicle Arrivals over Time')
plt.show()

# 将数据记录到文件中
with open('arrival_data.txt', 'w') as file:
    for time, arrival in zip(times, arrivals):
        file.write(f'{time} {arrival}\n')

