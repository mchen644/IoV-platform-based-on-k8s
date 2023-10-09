import matplotlib.pyplot as plt
import numpy as np
from functionality import exploration
from DDPG import DDPG
from env import Env
import concurrent.futures
import requests
from functionality import get_usage, get_state, send_action
import time

# States

# Actions

# Rewards




#####################  hyper parameters  ####################
episodes = 10 # Days
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
resource_variance = 1


if __name__ == "__main__":
    # Obtain initial environment state

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results = executor.map(get_usage, urls)

    initial_states = get_state(urls)

    print(initial_states)
    state_dim = len(initial_states)

    # 每个action只对应一个user，否则分配的资源维度和offload维度会发生变化，神经网络无法训练
    resource_dim = num_resource_type 
    offload_dim = 1

    # DDPG Model
    ddpg = DDPG(state_dim, resource_dim, offload_dim)
    num_users = 0
    episode_rewards = []
    var_rewards = []
    var_counter = 0

    for episode in episodes:
        
        arrivals = np.random.poisson(arrival_rate, len(time_sequence))
        episode_rewards.append(0)
        
        for i in range(len(arrival_rate)):
            # Simulate the time fly which corresponds to the real env: 0.1 hour
            time.sleep(time_interval)
            if arrival_rate[i] == 0:
                continue
            num_users = arrival_rate[i]
            
            # 这里可以考虑多线程
            for j in range(len(num_users)):
                # 先获得当前的State
                state = get_state(urls)
                
                action = ddpg.choose_action(state)

                # add randomness to action selection for exploration
                action = exploration(action, resource_dim, resource_variance)

                # TODO: Action should be sent back to the scheduler for scheduling
                # return rewards
                reward = send_action(action, send_action_URL_path)

                print(f"episode = {episode}, time = {time_sequence[i]}, \
                       user_id = {i}, when take action = {action}, reward = {reward}\n")
                
                state_ = get_state(urls)
                ddpg.store_transition(state, action, reward, state_)

                if ddpg.memory_count == ddpg.memory_capacity:
                    print("start learning")
                if ddpg.memory_count > ddpg.memory_capacity:
                    ddpg.learn()
                    if CHANGE:
                        resource_variance *= .99999
                # replace state
                episode_rewards[episode] += reward

            if i == len(arrival_rate)-1:
                var_rewards.append(episode_rewards[episode])
                 # variation change
                if var_counter >= var_change_check_episode and np.mean(var_reward[-var_change_check_episode:]) >= max_rewards:
                    CHANGE = True
                    var_counter = 0
                    max_rewards = np.mean(var_reward[-var_change_check_episode:])
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1



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
