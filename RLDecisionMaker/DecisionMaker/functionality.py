from http.server import HTTPServer
import numpy as np
import concurrent.futures
import requests
import json

map_node_num = {"k8s-as1":0, "controller":1, "gpu1": 2}
map_num_node = {0: "k8s-as1", 1: "controller", 2: "gpu1"}

# def explore_prob(a, epsilon=0.1, prob_var=0.01):
#     min_value = 0.0
#     max_value = 1.0
#     for i in range(2, 5):
#         if np.random.rand() < epsilon:
#             a[i] = np.random.uniform(min_value, max_value)
#         else:
#             a[i] = np.clip(np.random.normal(a[i], prob_var * 2), min_value, max_value)
#     return a

# 基本理念就是生成一个扰动向量，这个向量满足其元素和为0，并且使用一个在某个范围内取值的随机数来为元素赋值。然后将这个扰动向量加到原始向量之上，生成新的向量。这样，新的向量就相比原向量进行了一定程度的扰动，但是因为扰动向量的元素和为0，所以新向量的元素和仍然是1。
# 另外，为了保证结果向量的元素在[0, 1]范围内，这个代码加入了一些额外的逻辑：如果添加扰动向量后，某个元素超出了这个范围，那么就将这个元素剪裁到这个范围内。然后，最后一个步骤是重新标准化结果向量，保证其元素和为1。
# 这个代码的工作原理实际上是基于向量和随机数的基本概念，以及向量加法和Scala的基本运算规则，算不上复杂的数学原理，但是确实能够对原向量进行一定程度的扰动，并保证新向量的元素和仍然为1。

def explore_prob(a, episode, step_size=0.1, epsilon=1):
    epsilon = max(0.1, epsilon*(1-episode*step_size))
    # 首先，检查输入值是否满足条件
    if not np.isclose(sum(a[2:]), 1):
        raise ValueError("输入的三个数之和应为1")

    # 生成两个随机数
    r1 = np.random.uniform(-epsilon, epsilon)
    r2 = np.random.uniform(-epsilon, epsilon)

    # 生成第三个随机数，确保和为 0
    r3 = -r1 - r2

    # 生成扰动向量
    perturbation = np.array([r1, r2, r3])

    # 生成扰动后结果向量
    output_values = a[2:] + perturbation

    # 额外步骤: 对可能超出[0, 1]范围的值进行处理，这种情况在epsilon值较大时可能出现
    output_values = np.clip(output_values, 0, 1)

    # 如果剪裁后的和不再为1，需要重新标准化以满足要求
    if not np.isclose(sum(output_values), 1):
        output_values /= sum(output_values)

    return np.concatenate((a[:2], output_values))

def exploration(a, r_var, episode, CPU_remain, GPU_remain, offload_num, epsilon=0.1, max_episode=10, is_min_resource=False):
    # CPU resource exploration
    # lower_bound = max(100, a[0] - r_var)  # 下限为100或当前值减去r_var
    # upper_bound = min(CPU_remain[offload_num], a[0] + r_var)  # 上限为CPU_remain[offload_num]或当前值加上r_var
    # a[0] = np.random.uniform(lower_bound, upper_bound)
    min_cpu = 250.0  # 设定的启动pod所需的最小CPU资源
    min_gpu = 20.0  # 设定的最小GPU资源 启动det容器也需要一定的GPU资源好像
    if is_min_resource:
        a[0] = min_cpu
        a[1] = min_gpu
        return a
    
    max_cpu = CPU_remain[offload_num] * (0.5+0.5*episode / max_episode) if episode <= max_episode else CPU_remain[offload_num]
    max_gpu = GPU_remain[0] * (0.5 + 0.5*episode / max_episode) if episode <= max_episode else GPU_remain[0]

    # CPU resource exploration
    if np.random.rand() < epsilon:
        a[0] = max(min_cpu, np.random.uniform(min_cpu, max_cpu))
    else:
        a[0] = max(min_cpu, np.clip(np.random.normal(a[1], r_var * 2), 0, 1) * max_cpu)

    # GPU resource exploration
    if np.random.rand() < epsilon:
        a[1] = max(min_gpu, np.random.uniform(min_gpu, max_gpu))
    else:
        a[1] = max(min_gpu, np.clip(np.random.normal(a[1], r_var * 2), 0, 1) * max_gpu)
        
    return a

def select_one_offloading(resource_dim, action):
    import math
    prob_weights = action[resource_dim:]
    prob_weights[0] = math.trunc(prob_weights[0] * 1000) / 1000
    prob_weights[1] = math.trunc(prob_weights[1] * 1000) / 1000
    prob_weights[-1] = 1-prob_weights[0]-prob_weights[1]
    
    offloading_num = np.random.choice(range(len(prob_weights)), p=prob_weights.ravel())  # select action w.r.t the actions prob
    return offloading_num

def get_usage(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            cpu_usage = data["CPU"]
            gpu_usage = data["GPU"]
            print(f"{url[7:20]} CPU Usage:", cpu_usage)
            print(f"{url[7:20]} GPU Usage:", gpu_usage)
            return cpu_usage, gpu_usage
        else:
            print("Request failed with status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)

# def get_state(urls):
#     results = []
#     for url in urls:
#         result = get_usage(url)
#         results.append(result)

#     states = []
#     for result in results:
#         if result:
#             cpu_usage, gpu_usage = result
#             states.append(float(cpu_usage))
#             states.append(float(gpu_usage))
#     return states

import json
import fcntl
import os

def save_data(s, a, r, s_, filename):
    data = {'s': s.tolist(), 'a': a.tolist(), 'r': r, 's_': s_.tolist()}
    with open(filename, 'a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)  # 加锁
        json.dump(data, file)
        file.write('\n')  # 添加换行符以区分每个四个值的记录
        fcntl.flock(file, fcntl.LOCK_UN)  # 解锁

def init_memory(filename, memory_capacity, state_dim, action_dim):
    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist")
        return np.zeros((memory_capacity, state_dim * 2 + action_dim + 1),
                               dtype=np.float32), 0
    data_list = load_data(filename)
    memory_count = min(memory_capacity, len(data_list))
    memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1), dtype=np.float32)
    for i, data in enumerate(data_list):
        s = data['s']
        a = data['a']
        r = data['r']
        s_ = data['s_']
        # 将s、a、r、s_的值放入memory的每一行
        memory[i%memory_capacity] = np.array(s+a+[r]+s_)
    print(memory)
    return memory, memory_count

def load_data(filename):
    data_list = []
    with open(filename, 'r') as file:
        fcntl.flock(file, fcntl.LOCK_SH)  # 加共享锁
        lines = file.readlines()
        for line in lines:
            data = json.loads(line.strip())  # 去除换行符并解析JSON
            data_list.append(data)
        fcntl.flock(file, fcntl.LOCK_UN)  # 解锁
    return data_list

def get_num_pods():
    # 发起GET请求
    response = requests.get('http://192.168.1.101:8082/monitorPods')

    # 检查响应状态码
    if response.status_code == 200:
        # 解析JSON数据
        data = response.json()
    else:
        print('请求失败')
    return data

def get_total_num_pods():
    data = get_num_pods()
    return sum(data.values())

def get_state(CPU_remain, GPU_remain, queue):
    num_pods = get_num_pods()
    state = CPU_remain + GPU_remain
    state.append(num_pods['k8s-as1'])
    state.append(num_pods['controller'])
    state.append(num_pods['gpu1'])
    state.append(queue)
    return state

def run_http_server(host, port, RequestHandler):
    # 创建服务器实例，并指定请求处理程序
    server = HTTPServer((host, port), RequestHandler)
    print(f"Server running on {host}:{port}")

    # 启动服务器
    server.serve_forever()

def send_action(action, url) -> float:
    alpha = 0.7
    beta = 0.3
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            cpu_usage = data["CPU"]
            gpu_usage = data["GPU"]
            print(f"{url[7:20]} CPU Usage:", cpu_usage)
            print(f"{url[7:20]} GPU Usage:", gpu_usage)
            return alpha * avg_state - beta * latency
        else:
            print("Request failed with status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)