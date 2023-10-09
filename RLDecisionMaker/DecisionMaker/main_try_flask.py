import matplotlib.pyplot as plt
import numpy as np
from functionality import *
from DDPG import DDPG
from env import Env
import concurrent.futures
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import threading
from threading import Condition, Semaphore

#####################  hyper parameters  ####################
episodes = 10 # Days
num_episodes = 10
num_steps = 304
state_dim = 8
resource_dim = 2
offload_dim = 3
arrival_rate = 0.1 # cars arrival rate
time_sequence = np.arange(0, 24, 0.1) # Generate time sequence
time_interval = 0.1 # TODO:可能需要根据处理任务真实的延迟来考虑应该设置多少interval
var_change_check_episode = 4
CHANGE = False
num_resource_type = 2 # CPU, GPU
urls = [
    "http://192.168.1.101:8082/monitor",
    "http://192.168.1.100:8082/monitor",
    "http://192.168.1.106:8082/monitor"
    ]
send_action_URL_path = "http://192.168.1.101:8081/testCompleteTask"
resource_var = 10
prob_var = 0.01
CPU_bound = 1000.0 # CPU_bound for each node
GPU_bound = 100.0 # GPU_bound for each node
resource_bound = [CPU_bound] + [GPU_bound]
host = 'localhost'
port = 8083
filename = "train_data.json"
epsilon = 0.1
app = Flask(__name__)
# If received request for training, stop other threads dealing with other requests
# When obtaining s, a, r, s_, notify all
# global_condition = Condition() 
conditions = {'updateResource': Condition(), 'updateQueue': Condition(), 'queryActions': Condition()}
train_in_progress = False
read_in_progress = False
# train_in_progress_semaphore = Semaphore(10)  # 例如最大并行请求数为10

class ThreadedServer:
    ddpg = DDPG(state_dim, resource_dim, offload_dim)
    resource_explore_var = 1
    ep_reward = [0] # For plotting
    step_counter = 0
    episode_counter = 0
    resource_var_list = [] # For plotting
    resource_var = resource_var
    var_counter = 0
    max_rewards = 0
    var_reward = []
    CPU_remain = [CPU_bound]*3
    GPU_remain = [GPU_bound]
    num_pods = [0]*3
    state = CPU_remain + GPU_remain + num_pods
    state_ = []
    lock = threading.Lock()
    step_counter_lock = threading.Lock()
    check_resource_lock = threading.Lock()
    update_resource_lock = threading.Lock()
    update_queue_lock = threading.Lock()
    ep_reward_lock = threading.Lock()
    learn_lock = threading.Lock()
    train_lock = threading.Lock()
    queue = 0
    event = threading.Event()
    cond = threading.Condition()
    semaphore = threading.Semaphore(2)

@staticmethod
@app.route('/queryActions', methods=['GET'])
def handle_query_actions():
    print("--------------One request is querying action--------------------")
    # 先获得当前的State
    with conditions["queryActions"]:
        while train_in_progress:
            conditions["queryActions"].wait()
        with ThreadedServer.update_queue_lock:
            ThreadedServer.queue += 1

    while True:
        with ThreadedServer.check_resource_lock:
            state = np.array(get_state(ThreadedServer.CPU_remain, ThreadedServer.GPU_remain, ThreadedServer.queue))  # pods_num通过发请求得到
            action = np.array(ThreadedServer.ddpg.choose_action(state))
            print("Before explore_prob: ", action)
            action = explore_prob(action, epsilon, prob_var)
            print("After explore_prob: ", action)
            offload_num = select_one_offloading(resource_dim, action)
            offloading = map_num_node[offload_num]
            action = exploration(action, resource_var, ThreadedServer.episode_counter,  ThreadedServer.CPU_remain, ThreadedServer.GPU_remain, offload_num, epsilon=epsilon, max_episode=num_episodes)
            action_offloading = action[:2].tolist() + [offloading]
            if int(action_offloading[0]) >= ThreadedServer.CPU_remain[offload_num] or int(action_offloading[1]) >= ThreadedServer.GPU_remain[0]:
                print("Resource Insufficient!!!!!!!!! Waiting in queue for resource update !!!")
                with ThreadedServer.cond:
                    ThreadedServer.cond.wait()
                print("Resource have been updated, Try querying action again!!!!")
                state = np.array(get_state(ThreadedServer.CPU_remain, ThreadedServer.GPU_remain, ThreadedServer.queue))
                action = np.array(ThreadedServer.ddpg.choose_action(state))
                offload_num = select_one_offloading(resource_dim, action)
                offloading = map_num_node[offload_num]
                action = exploration(action, resource_var, ThreadedServer.episode_counter,  ThreadedServer.CPU_remain, ThreadedServer.GPU_remain, offload_num, epsilon=epsilon, max_episode=num_episodes, is_min_resource=True)
                action_offloading = action[:2].tolist() + [offloading]
            else:
                print("State: ", ' '.join([f'{x:.8f}' for x in state[:4]]), state[4:-1], state[-1])
                print("Action:", ' '.join([f'{x:.8f}' for x in action]))
                print("Action_offloading:", action_offloading)
                response_data = {
                    "State": state.tolist(), 
                    "Action": action.tolist(),
                    "CPU": int(action_offloading[0]),
                    "GPU": int(action_offloading[1]),
                    "Offloading": action_offloading[2]
                }
                ThreadedServer.GPU_remain = [ThreadedServer.GPU_remain[0] - action[1]]
                ThreadedServer.CPU_remain[map_node_num[action_offloading[2]]] = ThreadedServer.CPU_remain[map_node_num[action_offloading[2]]] - action[0]
                print(f"CPU_remain = {ThreadedServer.CPU_remain}, GPU_remain = {ThreadedServer.GPU_remain}")
                print("--------------Action Sending back! Ending querying action!--------------------")
                break

    # Action should be sent back to the scheduler for scheduling
    # action = [6000, 100, "k8s-as1"] # For testing, Note that in python:k8s-as1, but go: as1
    
    return jsonify(response_data)

@staticmethod
@app.route('/train', methods=['POST'])
def train():
    #print("Action received from scheduler: ", data)
    #print(f"CPU_remain = {ThreadedServer.CPU_remain}, GPU_remain = {ThreadedServer.GPU_remain}\n")
    global train_in_progress
    with ThreadedServer.train_lock:
        train_in_progress = True
        s_ = np.array(get_state(ThreadedServer.CPU_remain, ThreadedServer.GPU_remain, ThreadedServer.queue))
        r = 2-s_[4:7]-2*s_[8]
        train_in_progress = False
        for condition in conditions.values():
            with condition:
                condition.notify_all()  # 唤醒所有等待的路由
    #print("next state are: ", ' '.join([f'{x:.8f}' for x in s_]))
    data = request.get_json()  
    s = np.array(data.get('State'))
    # a = [data['CPU'], data['GPU'], data['Offloading']]
    a = np.array(data.get('Action'))
    save_data(s, a, r, s_, filename)
    # sum up the reward
    with ThreadedServer.ep_reward_lock:
        ThreadedServer.ep_reward[ThreadedServer.episode_counter] += r
        print(f"episode = {ThreadedServer.episode_counter+1}, step = {ThreadedServer.step_counter+1}, reward = {r}, total_episode_rewards={ThreadedServer.ep_reward[ThreadedServer.episode_counter]}")
        ThreadedServer.ddpg.store_transition(s, a, r, s_)
    
    # learn
    if ThreadedServer.ddpg.memory_count == ThreadedServer.ddpg.memory_capacity:
        print("start learning")
    if ThreadedServer.ddpg.memory_count > ThreadedServer.ddpg.memory_capacity:
        with ThreadedServer.learn_lock:
            ThreadedServer.ddpg.learn()
            if CHANGE:
                ThreadedServer.resource_var *= .5

    with ThreadedServer.step_counter_lock:
        ThreadedServer.step_counter += 1
        # In the end of the episode
        # 这边的代码待改进，因为原来使用的是var来进行循环(具体可以看一下源代码)，我是使用10个episodes来循环
        if ThreadedServer.step_counter == num_steps:
            ThreadedServer.var_reward.append(ThreadedServer.ep_reward[ThreadedServer.episode_counter])
            ThreadedServer.resource_var_list.append(ThreadedServer.resource_var)
            # variation change
            if ThreadedServer.var_counter >= var_change_check_episode and np.mean(ThreadedServer.var_reward[-var_change_check_episode:]) >= ThreadedServer.max_rewards:
                CHANGE = True
                ThreadedServer.var_counter = 0
                ThreadedServer.max_rewards = np.mean(ThreadedServer.var_reward[-var_change_check_episode:])
                ThreadedServer.var_reward = []
            else:
                CHANGE = False
                ThreadedServer.var_counter += 1

            print(f"Ending the {ThreadedServer.episode_counter+1}th episode, Beginning {ThreadedServer.episode_counter+2}th episode!!!!")
            ThreadedServer.episode_counter += 1
            with open("ep_reward.txt", "a") as file:
                file.write(str(ThreadedServer.ep_reward[ThreadedServer.episode_counter-1]) + "\n")
            ThreadedServer.ep_reward.append(0)
            ThreadedServer.step_counter = 0

    return "OK"

@staticmethod
@app.route('/updateResource', methods=['POST'])
def updateResource():
    data = request.get_json()  
    task_type = data.get('TaskType')
    GPU = data.get('GPU')
    Offloading = data.get('Offloading')
    CPU = data.get('CPU')
    with ThreadedServer.update_resource_lock:
        if task_type == "det":
            ThreadedServer.GPU_remain[0] += GPU
        elif task_type == "fusion":
            fusionNodeName = Offloading
            if fusionNodeName == "as1":
                fusionNodeName = "k8s-as1"
            ThreadedServer.CPU_remain[map_node_num[fusionNodeName]] += CPU
        print("-----------------------------update Resources successfully↓-----------------------------")
        print("Finishing task type: ", task_type)
        print(f"CPU_remain = {ThreadedServer.CPU_remain}, GPU_remain = {ThreadedServer.GPU_remain}")
        print("-----------------------------update Resources successfully↑-----------------------------")
        with ThreadedServer.cond:
            ThreadedServer.cond.notify_all()

    return "OK"

@staticmethod
@app.route('/updateQueue', methods=['POST'])
def updateQueue():
    with conditions['updateQueue']:
        while train_in_progress: # 存在 '/train' 请求正在处理
            conditions['updateQueue'].wait()
        with ThreadedServer.update_queue_lock:
            ThreadedServer.queue -= 1
            print("update queue num successfully")
            print("-----------------------------")
            print(f"queue num = ", ThreadedServer.queue)
            print("-----------------------------")
        # self.send_response(200)
        # self.end_headers()
    return "OK"

if __name__ == "__main__":
    # requestHandler = RequestHandler
    # run_http_server(host, port, requestHandler)
    app.run(threaded=True, port=port)