import numpy as np

from functionality import *
data_list = load_data("train_data.json")
data_length = len(data_list)
memory = np.zeros((30, 8 * 2 + 5 + 1), dtype=np.float32)
for i, data in enumerate(data_list):
    s = data['s']
    a = data['a']
    r = data['r']
    s_ = data['s_']
    # 将s、a、r、s_的值放入memory的每一行
    memory[i%30] = np.array(s+a+[r]+s_)

print(memory.size())