from my_agent import Agent
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.gamma = 0.95

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args, self.env)
            agents.append(agent)
        return agents

    def sample_episodes(self, agent_id):
        batch_obs = []
        batch_actions = []
        batch_rs = []
        for time_step in range(self.episode_limit):
            # reset the environment
            # self.env.render()
            reward_episode = []
            if time_step % self.episode_limit == 0:
                s = self.env.reset()

            u = []
            actions = []
            if agent_id == len(self.agents)-1:
                for i, agent in enumerate(self.agents):
                    if i == agent_id:
                        action = agent.policy.choose_action([s[i]], self.noise, self.epsilon)
                    else:
                        action = agent.policy.choose_action([s[i]], 0, 0)
                    u.append(action)
                    actions.append(action)
            else:
                for i, agent in enumerate(self.agents):
                    if i == len(self.agents)-1:
                        action = agent.policy.choose_action([s[i]], 0, 0)
                    else:
                        action = agent.policy.choose_action([s[i]], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            # for i in range(self.args.n_agents, self.args.n_players):
            #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            reward_episode.append(r[agent_id])
            reward_sum = 0.0
            discouted_sum_reward = np.zeros_like(reward_episode)
            for t in reversed(range(0, len(reward_episode))):
                reward_sum = reward_sum * self.gamma + reward_episode[t]
                discouted_sum_reward[t] = reward_sum
            #   ???????????????
            discouted_sum_reward = discouted_sum_reward.astype(np.float64)
            discouted_sum_reward -= np.mean(discouted_sum_reward)
            if np.std(discouted_sum_reward) != 0:
                discouted_sum_reward /= np.std(discouted_sum_reward)
            #   ??????????????????????????????????????????
            for t in range(len(reward_episode)):
                a = [discouted_sum_reward[t]] * 5
                batch_rs.append(a)
            batch_obs.append(s[agent_id].tolist())
            batch_actions.append(actions[agent_id].tolist())
            s = s_next
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
        return batch_obs, batch_actions, batch_rs

    def policy_train(self, agent_id):
        brain = self.agents[agent_id].policy
        # ??????1???episode
        train_obs, train_actions, train_rs = self.sample_episodes(agent_id)
        brain.train_step(train_obs, train_actions, train_rs)

    def policy_test(self):
        reward_sum = 0
        s = self.env.reset()
        for time_step in range(self.episode_limit):
            self.env.render()
            u = []
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.policy.choose_action([s[i]], 0, 0)
                u.append(action)
                actions.append(action)
            # for i in range(self.args.n_agents, self.args.n_players):
            #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            reward_sum += sum(r[:-1]) / (len(r) - 1)
        return reward_sum
        # env = self.env
        # observation = env.reset()[agent_id]
        # reward_sum = 0.0
        # env.render()
        # # ????????????????????????????????????
        # action = self.agents[agent_id].policy.choose_aciton([observation], 0, 0)
        # observation_, reward, done, info = env.step(action)
        # reward_sum += reward
        # return reward_sum

    def run(self):
        reward_sum_line = []
        for j in range(1000):
            for agent_id in range(self.args.n_agents):
                for i in range(50):
                    self.policy_train(agent_id)
                    if i % 25 == 0:
                        reward_sum = self.policy_test()
                        print("agent_id:", agent_id, "episode:", j, "step:", i, "reward_sum:", reward_sum)
                        reward_sum_line.append(reward_sum)
                        plt.figure()
                        plt.plot(range(len(reward_sum_line)), reward_sum_line)
                        plt.xlabel("testing number")
                        plt.ylabel("score")
                        plt.show()

