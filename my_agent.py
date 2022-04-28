import numpy as np
import torch
import os
from OptGradFP import Policy_Net


class Agent:
    def __init__(self, agent_id, args, env):
        self.args = args
        self.agent_id = agent_id
        self.env = env
        self.policy = Policy_Net(args, agent_id, env)

    # def learn(self, transitions, other_agents, agent):
    #     self.policy_train(transitions, other_agents, agent)

