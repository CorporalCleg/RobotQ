import random

import imageio
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm.notebook import tqdm


def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


class Policy:
    def __call__(Qtable, state):
        pass

    def update():
        None


class GreedyPolicy(Policy):
    def __init__(self):
        super().__init__()

    def __call__(self, Qtable, state):
        return np.argmax(Qtable[state][:])


class EpsilonGreedyPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.greedy = GreedyPolicy()

    def __call__(self, Qtable, state):
        random_num = np.random.binomial(1, 0.2)
        # if random_num > greater than epsilon --> exploitation
        if random_num:
            # Take the action with the highest value given a state
            # np.argmax can be useful here
            action = self.greedy(Qtable, state)
            # else --> exploration
        else:
            action = env.action_space.sample()
        return action


def softmax(seq):
    a = np.exp(seq)
    return a / (1e-10 + a.sum())


class SoftmaxPolicy(Policy):
    def __init__(self):
        super().__init__()

    def __call__(self, Qtable, state):  # take action with prob. of it's value
        return np.random.choice([0, 1, 2, 3], p=softmax(Qtable[state][:]))


class Qlearning:
    def __init__(self, env, train_policy):
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.Qtable = initialize_q_table(self.nS, self.nA)
        self.train_policy = train_policy

    def train(self, n_training_episodes=10, max_steps=10, lr=0.7, gamma=0.99):
        for episode in tqdm(range(n_training_episodes)):
            # self.train_policy.step()
            state, info = env.reset()
            for step in range(max_steps):

                action = self.train_policy(self.Qtable, state)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated  # if we dont want to use next ep.

                td_error = (
                    reward
                    + gamma * np.max(self.Qtable[new_state])
                    - self.Qtable[state][action]
                )
                
                self.Qtable[state][action] = self.Qtable[state][action] + lr * td_error

                # If terminated or truncated finish the episode
                if terminated or truncated:
                    break

                # Our next state is the new state
                state = new_state
        return self.Qtable


class DoubleQlearning:
    def __init__(self, env, train_policy):
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.Qtable = initialize_q_table(self.nS, self.nA)
        self.train_policy = train_policy

    def train(self, n_training_episodes=10, max_steps=10, lr=0.7, gamma=0.99):
        Q1 = np.zeros_like(self.Qtable)
        Q2 = np.zeros_like(self.Qtable)

        for episode in tqdm(range(n_training_episodes)):
            # self.train_policy.step()
            state, info = self.env.reset()
            for step in range(max_steps):

                action = self.train_policy((Q1 + Q2) / 2.0, state)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated  # if we dont want to use next ep.

                if np.random.randint(2):

                    argmax = np.argmax(Q1[state][:])
                    td_error = (
                        reward + gamma * Q2[new_state][argmax] - Q1[state][action]
                    )
                    Q1[state][action] = Q1[state][action] + lr * td_error

                else:
                    argmax = np.argmax(Q2[state][:])
                    td_error = (
                        reward + gamma * Q1[new_state][argmax] - Q2[state][action]
                    )
                    Q2[state][action] = Q2[state][action] + lr * td_error

                # If terminated or truncated finish the episode
                if terminated or truncated:
                    break

                # Our next state is the new state
                state = new_state

        self.Qtable = (Q1 + Q2) / 2.0
        return self.Qtable
