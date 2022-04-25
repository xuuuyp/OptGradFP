import math
import random
import numpy as np

class OptGradFP:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        self.ep_max = 400
        self.ns = 25
        self.nb = 250
        self.E = 10000
        self.lr_d = 5 * math.exp() - 4
        self.lr_o = 5 * math.exp() - 4
        self.beta_d = 9 / self.ep_max
        self.beta_o = 9 / self.ep_max

        self.wd = random.randint()
        self.wo = random.randint()
        self.mem_cnt = 0
        # TODO(xu) state_dim
        self.state_memory = np.zeros((self.E, self.state_dim))
        self.action_d_memory = np.zeros((self.E, ))
        self.action_o_memory = np.zeros((self.E, ))

    def store_transition(self, state, action_d, action_o):
        mem_idx = self.mem_cnt % self.E
        self.state_memory[mem_idx] = state
        self.action_d_memory[mem_idx] = action_d
        self.action_o_memory[mem_idx] = action_o
        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.E, self.mem_cnt)
        # Todo batch_size是不是nd
        batch = np.random.choice(mem_len, self.nb, replace=True)

        states = self.state_memory[batch]
        action_d = self.action_d_memory[batch]
        action_o = self.action_o_memory[batch]
        return states, action_d, action_o

    def ready(self):
        return self.mem_cnt > self.nb

    # Todo 主要改这两个，经验回放不归这里管
    def train(self, transitions, other_agents):

        s, a_d, a_o = [], [], []
        for agent_id in range(self.args.n_agents):
            s.append(transitions[0][agent_id])
            a_d.append(transitions[1][agent_id])
            a_o.append(transitions[2][agent_id])


        pass

    def project_act(self, s, a_d, agent_id):


    def act(self, s):
        pass


