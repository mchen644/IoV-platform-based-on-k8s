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

# States

# Actions

# Rewards

#####################  hyper parameters  ####################
episodes = 10 # Days
num_episodes = 10
num_steps = 371 
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
CPU_bound = 1000.0 # CPU_bound for each node
GPU_bound = 100.0 # GPU_bound for each node
resource_bound = [CPU_bound] + [GPU_bound]
# 设置服务器的主机名和端口
host = 'localhost'
port = 8083
filename = "train_data.json"
epsilon = 0.1

# 创建自定义的请求处理程序
class RequestHandler(BaseHTTPRequestHandler):
    #     super().__init__(request, client_address, server)
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
    queue = 0
    event = threading.Event()
    cond = threading.Condition()
    semaphore = threading.Semaphore(2)

     # 处理POST请求
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # 检查 post_data 是否为有效的 JSON
        if post_data:
            try:
                data = json.loads(post_data)
            except json.decoder.JSONDecodeError as e:
                # 处理无效 JSON 数据的错误
                print("Invalid JSON data:", e)
                data = None
        else:
            data = None

        # 根据请求路径执行不同的处理逻辑
        if self.path == "/train":
            # thread = threading.Thread(target=self.train, args=(data,))
            # thread.start()
            self.train(data)
        elif self.path == "/updateResource":
            # thread = threading.Thread(target=self.updateResource, args=(data,))
            # thread.start()
            self.updateResource(data)
        elif self.path == "/updateQueue":
            # thread = threading.Thread(target=self.updateQueue)
            # thread.start()
            self.updateQueue()
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def updateQueue(self):
        with self.__class__.lock:
            self.__class__.queue -= 1
        print("update queue num successfully")
        print("-----------------------------")
        print(f"queue num = ", self.__class__.queue)
        print("-----------------------------")
        self.send_response(200)
        # self.end_headers()
        return 

    # 处理GET请求
    def do_GET(self):
        if self.path == "/queryActions":
            thread = threading.Thread(target=self.handle_query_actions)
            thread.start()
            thread.join()
            # self.handle_query_actions()
        elif self.path == "/test":
            self.handle_test()
        else:
            self.send_response(404)
            self.end_headers()
    
    def updateResource(self, data):
        print("Finishing task type: ", data['TaskType'])
        with self.__class__.lock:
            if data["TaskType"] == "det":
                self.__class__.GPU_remain[0] += data["GPU"]
            elif data["TaskType"] == "fusion":
                fusionNodeName = data["Offloading"]
                if fusionNodeName == "as1":
                    fusionNodeName = "k8s-as1"
                self.__class__.CPU_remain[map_node_num[fusionNodeName]] += data["CPU"]
        print("-----------------------------update Resources successfully↓-----------------------------")
        print(f"CPU_remain = {self.__class__.CPU_remain}, GPU_remain = {self.__class__.GPU_remain}\n")
        print("-----------------------------update Resources successfully↑-----------------------------")
        with self.__class__.cond:
            self.__class__.cond.notify_all()
        self.send_response(200)
        # self.end_headers() 
        return 
    
    def train(self, data):
        self.__class__.step_counter += 1
        # 先从scheduler发送的json数据中得到s a r s_
         # 解析接收到的JSON数据
        s = np.array(data['State'])
        # a = [data['CPU'], data['GPU'], data['Offloading']]
        a = np.array(data['Action'])
        print("Action received from scheduler: ", data)
        print(f"CPU_remain = {self.__class__.CPU_remain}, GPU_remain = {self.__class__.GPU_remain}\n")
        s_ = np.array(get_state(self.__class__.CPU_remain, self.__class__.GPU_remain, self.__class__.queue))
        r = 2-get_total_num_pods()-2*self.__class__.queue
        print("next state are: ", ' '.join([f'{x:.8f}' for x in s_]))
        save_data(s, a, r, s_, filename)
        # sum up the reward
        self.__class__.ep_reward[self.__class__.episode_counter] += r
        print(f"episode = {self.__class__.episode_counter}, step = {self.__class__.step_counter}, reward = {r}, total_episode_rewards={self.__class__.ep_reward[self.__class__.episode_counter]}")
        # while(True):
        #     time.sleep(1)
        self.__class__.ddpg.store_transition(s, a, r, s_)    
        # learn
        if self.__class__.ddpg.memory_count == self.__class__.ddpg.memory_capacity:
            print("start learning")
        if self.__class__.ddpg.memory_count > self.__class__.ddpg.memory_capacity:
            self.__class__.ddpg.learn()
            if CHANGE:
                self.__class__.resource_var *= .5

        # In the end of the episode
        # 这边的代码待改进，因为原来使用的是var来进行循环(具体可以看一下源代码)，我是使用10个episodes来循环
        if self.__class__.step_counter == num_steps - 1:
            self.__class__.var_reward.append(self.__class__.ep_reward[self.__class__.episode_counter])
            self.__class__.resource_var_list.append(self.__class__.resource_var)
            # variation change
            if self.__class__.var_counter >= var_change_check_episode and np.mean(self.__class__.var_reward[-var_change_check_episode:]) >= self.__class__.max_rewards:
                CHANGE = True
                self.__class__.var_counter = 0
                self.__class__.max_rewards = np.mean(self.__class__.var_reward[-var_change_check_episode:])
                self.__class__.var_reward = []
            else:
                CHANGE = False
                self.__class__.var_counter += 1
            self.__class__.episode_counter += 1
            print(f"Ending the %dth episode, Beginning new episode", self.__class__.episode_counter)
            with open("ep_reward.txt", "w") as file:
                file.write(str(self.__class__.ep_reward[self.__class__.episode_counter-1]))
            self.__class__.ep_reward.append(0)
            self.__class__.step_counter = 0
        self.send_response(200)
        # self.end_headers() 
        return 

    def handle_query_actions(self):
        print("--------------One request is querying action--------------------")
        # 先获得当前的State
        with self.__class__.lock:
            self.__class__.queue += 1
        state = np.array(get_state(self.__class__.CPU_remain, self.__class__.GPU_remain, self.__class__.queue))  # pods_num通过发请求得到
        action =  np.array(self.__class__.ddpg.choose_action(state))
        offload_num = select_one_offloading(resource_dim, action)
        offloading = map_num_node[offload_num]
        # print("Offloading to: ", offloading)
        # print("before exploration:", ' '.join([f'{x:.8f}' for x in action]))
        # add randomness to action selection for exploration
        action = exploration(action, resource_var, self.__class__.episode_counter,  self.__class__.CPU_remain, self.__class__.GPU_remain, offload_num, epsilon=epsilon, max_episode=num_episodes)
        # print("after exploration:", action)
        # print("after exploration:", ' '.join([f'{x:.8f}' for x in action]))
        
        # while(True):
        #     time.sleep(1)
        action_offloading = action[:2].tolist() + [offloading]

        while int(action_offloading[0]) > self.__class__.CPU_remain[offload_num] or int(action_offloading[1]) > self.__class__.GPU_remain[0]:
            print("Resource Insufficient!!!!!!!!! Waiting in queue for resource update !!!")
            with self.__class__.cond:
                self.__class__.cond.wait()  # 阻塞，等待条件变量被通知
            print("Resource have been updated, Try querying action again!!!!")
            state = np.array(get_state(self.__class__.CPU_remain, self.__class__.GPU_remain, self.__class__.queue))
            action = np.array(self.__class__.ddpg.choose_action(state))
            offload_num = select_one_offloading(resource_dim, action)
            offloading = map_num_node[offload_num]
            # print("Offloading to: ", offloading)
            # print("before exploration:", ' '.join([f'{x:.8f}' for x in action]))
            action = exploration(action, resource_var, self.__class__.episode_counter,  self.__class__.CPU_remain, self.__class__.GPU_remain, offload_num, epsilon=epsilon, max_episode=num_episodes)
            action_offloading = action[:2].tolist() + [offloading]
            # print("after exploration:", ' '.join([f'{x:.8f}' for x in action]))
        print("State: ", ' '.join([f'{x:.8f}' for x in state]))
        print("Action:", ' '.join([f'{x:.8f}' for x in action]))
        print("Action_offloading:", action_offloading)

        # Action should be sent back to the scheduler for scheduling
        # action = [6000, 100, "k8s-as1"] # For testing, Note that in python:k8s-as1, but go: as1
        response_data = {
            "State": state.tolist(), 
            "Action": action.tolist(),
            "CPU": int(action_offloading[0]),
            "GPU": int(action_offloading[1]),
            "Offloading": action_offloading[2]
        }

        with self.__class__.lock:
            self.__class__.GPU_remain = [self.__class__.GPU_remain[0] - action[1]]
            self.__class__.CPU_remain[map_node_num[action_offloading[2]]] = self.__class__.CPU_remain[map_node_num[action_offloading[2]]] - action[0]
        print(f"CPU_remain = {self.__class__.CPU_remain}, GPU_remain = {self.__class__.GPU_remain}\n")
        print("--------------Action Sending back! Ending querying action!--------------------")
        
        response_json = json.dumps(response_data)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        
        self.wfile.write(response_json.encode())
       
        return
        
    #     # return rewards
    #     reward = send_action(action, send_action_URL_path)

    #     print(f"episode = {episode}, time = {time_sequence[i]}, \
    #         user_id = {i}, when take action = {action}, reward = {reward}\n")
        
    #     state_ = get_state(urls)
    #     ddpg.store_transition(state, action, reward, state_)

    #     if ddpg.memory_count == ddpg.memory_capacity:
    #         print("start learning")
    #     if ddpg.memory_count > ddpg.memory_capacity:
    #         ddpg.learn()
    #         if CHANGE:
    #             resource_variance *= .99999

    #     self.episode_rewards[self.episode_counter] += reward

    # if i == len(arrival_rate)-1:
    #     var_rewards.append(episode_rewards[episode])
    #     # variation change
    #     if var_counter >= var_change_check_episode and np.mean(var_reward[-var_change_check_episode:]) >= max_rewards:
    #         CHANGE = True
    #         var_counter = 0
    #         max_rewards = np.mean(var_reward[-var_change_check_episode:])
    #         var_reward = []
    #     else:
    #         CHANGE = False
    #         var_counter += 1
    

    
    def handle_test(self):
        # 发送响应状态码
        self.send_response(200)

        # 设置响应头的Content-type为application/json
        self.send_header("Content-type", "application/json")
        self.end_headers()

        # 创建一个结构体（字典）
        response_data = {
            "message": "Handling /queryActions route",
            "data": [1, 2, 3, 4, 5]
        }

        # 将结构体转换为JSON格式
        response_json = json.dumps(response_data)

        # 发送响应内容
        self.wfile.write(response_json.encode())

        return

    



if __name__ == "__main__":
    # 跑http server：每收到一个request，都会有一条训练数据存入memory库中
    # MIRAS里面使用的是1000步训练一次，拿这1000次用于训练，每25步reset一次env
    # project中每3000 steps重置一次env
    # 先定义state和action

    # state: CPU_node0, CPU_node1, CPU_node6, GPU_node6, 
    # action: fusion task offloading + CPU limit + GPU limit
    


    # 如果GPU分配很多的时候，还是只需要很少的CPU资源就能将slam跑起来 怎么办？
    # 这种情况下就需要控制分配给fusion task的cpu limit，否则瓶颈都在GPU上
    # resource constraint: As we deal with resource allocation problem under constraints, it’s important to find
    # the correct constraints for the microservice systems. A good
    # constraint means that we don’t have redundant resources so
    # that good resource allocation policies are unnecessary, and
    # also resources should be sufficient so that feasible resource
    # allocation solutions can be found.
    
    
    # 所以关于resource constraint这块，可以限制GPU limit constraint 为 100%
    # CPU limit constraint 为 每个node上最多能分配的cpu limit, 先设置为1000试试

    requestHandler = RequestHandler
    
    # reward: time window开始时候的task量-time window结束时的task量 + time window下的平均CPU， GPU使用率
    run_http_server(host, port, requestHandler)
    
    














    # Obtain initial environment state

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results = executor.map(get_usage, urls)

    # Reward corresponding to each episode
    # episode_rewards = []
    
    # # Environment:
    # env = Env()
    # resource_dim =     # Consider CPU, GPU for now TODO: Consider more resource: CPU, GPU
    # offload_dim = 3     # self.edge_num * len(self.U)
    # action_dim = resource_dim + offload_dim
    # state_dim =         # TODO: state_dim
    # resource_bound =           # TODO: resource_bound

    # # DDPG Model
    # ddpg = DDPG(state_dim, resource_dim, offload_dim, resource_bound)

    # # Reward corresponding to each episode
    # episode_rewards = []

    # # set up exploration variance(resource)
    # resource_variance = 1
    # max_rewards = 0
    # var_counter = 0
    # var_rewards = []


    # episode = 0

    # while var_counter < episodes:
    #     # reset state
    #     S =

    #     # 初始化当前episode的reward为0
    #     episode_rewards.append(0)

    #     # 开始每个episode的每一个step
    #     for step in range(steps):
    #         # DDPG
    #         # 1. choose action according to the current state
    #         a = ddpg.choose_action(S)
    #         # 2. exploration
    #         a = exploration(a, resource_dim, resource_variance, r_bound)
    #         # 3. transition
    #         S_, r = env.ddpg_step_forward(a, resource_dim)
    #         # 4. store the transition parameter
    #         ddpg.store_transition(S, a, r / 10, S_) # TODO: Why r/10?

    #         # 5. learn
    #         if ddpg.memory_count == ddpg.memory_capacity:
    #             print("ddpg model start to learn from the memory_capacity")
    #         if ddpg.memory_count > ddpg.memory_capacity:
    #             ddpg.learn()
    #             if CHANGE:
    #                 resource_variance *= .99999

    #         # 6. update the current state
    #         S = S_

    #         # 7. add reward obtained to each episode's total reward
    #         episode_rewards[episode] += r

    #         # 8. last step of the episode:
    #         if step == steps - 1:
    #             print(f"Episode:{episode}, Reward: {episode_rewards[episode]}")

    #             # 9. Find out if we want to do more exploration
    #             # Check when var_counter > var_change_check_episode
    #             # if the newly obtained average reward is large enough
    #             # meaning that the exploration works
    #             # we may expand our exploration
    #             # Or we just increase var_counter and check every episode if we obtain more rewards
    #             # if not until var_counter > episodes, we stop learning
    #             if var_counter >= var_change_check_episode and np.mean(var_rewards[-var_change_check_episode:]) >= max_rewards:
    #                 CHANGE = True
    #                 var_counter = 0
    #                 max_rewards = np.mean(var_rewards[-var_change_check_episode:])
    #                 var_reward = []
    #             else:
    #                 CHANGE = False
    #                 var_counter += 1
    #     episode += 1

    # # plot the reward
    # fig_reward = plt.figure()
    # plt.plot([i + 1 for i in range(episode)], episode_rewards)
    # plt.xlabel("episode")
    # plt.ylabel("rewards")
    # fig_reward.savefig('rewards.png')
