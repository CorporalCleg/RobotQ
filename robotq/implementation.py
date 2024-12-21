import random

import wandb
import imageio
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm.notebook import tqdm


def initialize_q_table(state_space, action_space):  # use it to make Q-table
    Qtable = np.zeros((state_space, action_space))
    return Qtable


class Policy:
    """
    A policy template class that defines the interface for different policies.

    The policy class is responsible for selecting an action given a Q-table and a state.
    """

    def __call__(self, Qtable, state):
        """
        Selects an action given a Q-table and a state.

        Args:
        Qtable (numpy.array): The Q-table.
        state (int): The current state.

        Returns:
        int: The selected action.
        """
        pass

    def update(self):
        """
        Updates the policy.

        This method is currently a placeholder and does not perform any updates.
        """
        None


class GreedyPolicy(Policy):
    """
    A simple greedy policy that selects the action with the highest Q-value.

    Attributes:
    None
    """

    def __init__(self):
        """
        Initializes the GreedyPolicy class.

        Calls the Policy class constructor.
        """
        super().__init__()

    def __call__(self, Qtable, state):
        """
        Selects the action with the highest Q-value.

        Args:
        Qtable (numpy.array): The Q-table.
        state (int): The current state.

        Returns:
        int: The action with the highest Q-value.
        """
        return np.argmax(Qtable[state][:])


def softmax(seq):
    """
    Computes the softmax of a sequence.

    Args:
    seq (numpy.array): The input sequence.

    Returns:
    numpy.array: The softmax of the input sequence.
    """
    a = np.exp(seq)
    return a / (a.sum())


class SoftmaxPolicy(Policy):
    """
    A softmax policy that selects an action with probability proportional to its Q-value.

    Attributes:
    None
    """

    def __init__(self):
        """
        Initializes the SoftmaxPolicy class.

        Calls the Policy class constructor.
        """
        super().__init__()

    def __call__(self, Qtable, state):
        """
        Selects an action using the softmax strategy.

        Args:
        Qtable (numpy.array): The Q-table.
        state (int): The current state.

        Returns:
        int: The selected action.
        """
        return np.random.choice(list(range(len(Qtable[state]))), p=softmax(Qtable[state][:]))


class UCBPolicy(Policy):
    def __init__(self, c=1):
        super().__init__()
        self.e = 0  # Count of actions taken so far (episodes)
        self.c = c  # Exploration parameter
        self.N = None  # Count of times each action has been taken

    def __call__(self, Qtable, state):
        Q = Qtable[state, :]
        num_actions = len(Q)

        if self.N is None:
            self.N = np.zeros(num_actions)
        if self.e < num_actions:
            action = self.e
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                U = np.sqrt(self.c * np.log(max(1, self.e)) / (self.N + 1e-8))
            U[self.N == 0] = np.inf
            action = np.argmax(Q + U)
        self.N[action] += 1
        self.e += 1
        return action



class ThompsonSamplingPolicy(Policy):

    def __init__(self, alpha=1.0, beta=0.0):
        self.alpha = alpha
        self.beta = beta
        self.N = None

    def __call__(self, Qtable, state):
        Q = Qtable[state, :]
        if self.N is None:
            self.N = np.zeros(len(Q))
        samples = np.random.normal(
            loc=Q, scale=self.alpha/(np.sqrt(self.N) + self.beta))
        action = np.argmax(samples)

        self.N[action] += 1
        return action


class Qlearning:
    """
    Q-learning is a model-free reinforcement learning algorithm that learns to predict the expected return of an action in a given state.

    Attributes:
    env (gym.Env): The environment to train the agent in.
    train_policy (function): The policy to use for training.
    nS (int): The number of states in the environment.
    nA (int): The number of actions in the environment.
    Qtable (numpy.array): The Q-table to store the action values.
    """

    def __init__(self, env, train_policy, log=None):
        """
        Initializes the Q-learning agent.

        Args:
        env (gym.Env): The environment to train the agent in.
        train_policy (function): The policy to use for training.
        """
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.Qtable = initialize_q_table(self.nS, self.nA)
        self.train_policy = train_policy
        self.log = None
        if log:
            self.log = True
            wandb.login()

    def train(self, n_training_episodes=10, max_steps=10, lr=0.7, gamma=0.99):
        """
        Trains the Q-learning agent.

        Args:
        n_training_episodes (int, optional): The number of training episodes. Defaults to 10.
        max_steps (int, optional): The maximum number of steps per episode. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.7.
        gamma (float, optional): The discount factor. Defaults to 0.99.

        Returns:
        numpy.array: The trained Q-table.
        """

        if self.log:
            run = wandb.init(
                # Set the project where this run will be logged
                project="Q-learning",
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": lr,
                    "epochs": n_training_episodes,
                },
            )

        for episode in tqdm(range(n_training_episodes)):
            # self.train_policy.step()
            state, info = self.env.reset()
            if self.log:
                wandb.log({"Q[0]":self.Qtable[state][0],
                            "Q[1]":self.Qtable[state][1],
                            "Q[2]":self.Qtable[state][2],
                            "Q[3]":self.Qtable[state][3],
                            "Q[4]":self.Qtable[state][4],
                            "Q[5]":self.Qtable[state][5]
                        })
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

class DoubleQlearning(Qlearning):
    """
    Double Q-learning is a model-free reinforcement learning algorithm that combines
    the Q-learning updates from two Q-tables to reduce the overestimation bias.

    Attributes:
    env (gym.Env): The environment to train the agent in.
    train_policy (function): The policy to use for training.
    nS (int): The number of states in the environment.
    nA (int): The number of actions in the environment.
    Qtable1 (numpy.array): The first Q-table to store action values.
    Qtable2 (numpy.array): The second Q-table to store action values.
    """

    def __init__(self, env, train_policy):
        super().__init__(env, train_policy)
        self.Qtable1 = initialize_q_table(self.nS, self.nA)
        self.Qtable2 = initialize_q_table(self.nS, self.nA)

    def train(self, n_training_episodes=10, max_steps=10, lr=0.7, gamma=0.99):
        """
        Trains the Double Q-learning agent.

        Args:
        n_training_episodes (int, optional): The number of training episodes. Defaults to 10.
        max_steps (int, optional): The maximum number of steps per episode. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.7.
        gamma (float, optional): The discount factor. Defaults to 0.99.

        Returns:
        numpy.array: The trained Q-table.
        """
        for episode in tqdm(range(n_training_episodes)):
            state, info = self.env.reset()
            for step in range(max_steps):

                action = self.train_policy(self.Qtable1, state)

                # Select the table to update
                if np.random.rand() < 0.5:
                    Qtable = self.Qtable1
                else:
                    Qtable = self.Qtable2

                new_state, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated

                # Double Q-learning update
                other_table = self.Qtable2 if Qtable is self.Qtable1 else self.Qtable1
                td_error = (
                    reward
                    + gamma * np.max(other_table[new_state])
                    - Qtable[state][action]
                )

                Qtable[state][action] = Qtable[state][action] + lr * td_error

                if terminated or truncated:
                    break

                state = new_state
        return (self.Qtable1 + self.Qtable2) / 2.0


class DynaQ:
    """
    Dyna-Q is a model-based reinforcement learning algorithm that combines Q-learning with planning.

    Attributes:
    env (gym.Env): The environment to train the agent in.
    train_policy (function): The policy to use for training.
    nS (int): The number of states in the environment.
    nA (int): The number of actions in the environment.
    Qtable (numpy.array): The Q-table to store the action values.
    """

    def __init__(self, env, train_policy):
        """
        Initializes the Dyna-Q agent.

        Args:
        env (gym.Env): The environment to train the agent in.
        train_policy (function): The policy to use for training.
        """
        self.env = env
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n

        self.Qtable = np.zeros((self.nS, self.nA))
        self.train_policy = train_policy

    def train(
        self,
        n_training_episodes=10,
        max_steps=10,
        lr=0.7,
        gamma=0.99,
        n_planning=3,
        return_model=False,
    ):
        """
        Trains the Dyna-Q agent.

        Args:
        n_training_episodes (int, optional): The number of training episodes. Defaults to 10.
        max_steps (int, optional): The maximum number of steps per episode. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.7.
        gamma (float, optional): The discount factor. Defaults to 0.99.
        n_planning (int, optional): The number of planning steps. Defaults to 3.
        return_model (bool, optional): Whether to return the model. Defaults to False.

        Returns:
        numpy.array: The trained Q-table, or the Q-table and the model if return_model is True.
        """
        T_count = np.zeros(
            (self.nS, self.nA, self.nS), dtype=np.int32
        )  # store frequencies
        R_model = np.zeros(
            (self.nS, self.nA, self.nS), dtype=np.float64
        )  # store rewards

        for e in tqdm(range(n_training_episodes)):

            state, info = self.env.reset()

            for step in range(max_steps):

                action = self.train_policy(self.Qtable, state)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated  # if we dont want to use next ep.

                # If terminated or truncated finish the episode

                T_count[state, action, new_state] += 1  # learn our model
                r_diff = reward - R_model[state, action, new_state]
                R_model[state, action, new_state] += (
                    r_diff / T_count[state, action, new_state]
                )

                td_error = (
                    reward
                    + gamma * np.max(self.Qtable[new_state])
                    - self.Qtable[state][action]
                )
                self.Qtable[state][action] = self.Qtable[state][action] + lr * td_error

                # Our next state is the new state
                backup_state = new_state

                # utilize our model to learn agent
                for _ in range(
                    n_planning
                ):  # sample (state, action, new_state) x n and re-learn reward

                    if self.Qtable.sum() == 0:
                        break

                    visited_states = np.where(np.sum(T_count, axis=(1, 2)) > 0)[0]
                    state = np.random.choice(visited_states)

                    actions_taken = np.where(np.sum(T_count[state], axis=1) > 0)[0]
                    action = np.random.choice(actions_taken)

                    probs = T_count[state][action] / T_count[state][action].sum()

                    next_state = np.random.choice(np.arange(self.nS), size=1, p=probs)[
                        0
                    ]

                    reward = R_model[state][action][next_state]
                    td_target = reward + gamma * self.Qtable[next_state].max()

                    td_error = td_target - self.Qtable[state][action]

                    self.Qtable[state][action] = (
                        self.Qtable[state][action] + lr * td_error
                    )

                    state = backup_state

                if done:
                    break
        if return_model:
            return self.Qtable, R_model
        else:
            return self.Qtable