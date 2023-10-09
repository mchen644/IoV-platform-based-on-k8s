import numpy as np

#####################  hyper parameters  ####################
num_resource_type = 2  # CPUlimit, GPUlimit
num_nodes = 2  # fusion Node
num_users = 10  # TODO: How to define the number of the users

class Env():
    def __init__(self, num_edges, num_users):
        self.num_edges = num_edges
        self.num_users = num_users
        self.Users = []
        self.resources = np.zeros((num_resource_type*self.num_users))
        self.offloads = np.zeros(self.num_users)

    


    def reset(self):
        # Users:
        self.Users = []

    def ddpg_step_forward(self, a, resource_dim, ):

        self.resources = a[:resource_dim]   # resource dim = resource type * num_users
        offloading_base = resource_dim

        for user_id in range(self.num_users):
            prob_weights_offloading = a[offloading_base:offloading_base + self.num_edges]
            action = np.random.choice(range(len(prob_weights_offloading)), p=prob_weights_offloading.ravel())
            offloading_base += self.num_edges
            self.offloads[user_id] = action

