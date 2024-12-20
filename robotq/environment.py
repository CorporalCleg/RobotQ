import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import numpy as np
import cv2
from scipy import signal
import random

import imageio
from tqdm.notebook import tqdm

from utils import evaluate_agent, record_video

def generate_map(map_size=(40, 40), num_circles=10): # generate 100x100 map with 10 circles
    gen_map = np.zeros(map_size)

    # define 
    values, probs = [1, 2, 3, 4, 5], [0.4, 0.3, 0.2, 0.05, 0.05]
    for _ in range(num_circles):
        radius = np.random.choice(values, p=probs)
        cx, cy = np.random.randint(radius + 1, map_size[0] - radius - 1), np.random.randint(radius + 1, map_size[1] - radius - 1)
        gen_map = cv2.circle(gen_map, (cx, cy), radius, color=1, thickness=-1)

    return gen_map

def agent_image(): # generates images of 4 rotational states
    size = 11
    stick = np.zeros((4, size, size))

    stick[0, 0, size // 2] = 1
    stick[0, -1, size // 2] = 1

    stick[1, 0, 0] = 1
    stick[1, -1, -1] = 1

    stick[2, size // 2, 0] = 1
    stick[2, size // 2, -1] = 1

    stick[3, 0, -1] = 1
    stick[3, -1, 0] = 1

    return stick

def get_collision_map(g_map, stick): # return image, with "0" where space is free
    padding = 10

    c_image = [np.pad(g_map, ((padding, padding), (padding, padding)), 'constant', constant_values=1) for i in range(4)]
    c_image = [signal.convolve2d(c_image[i], stick[i,:,:], boundary='symm', mode='same')for i in range(4)]
    c_image = [c_image[i][padding:-padding, padding:-padding] for i in range(4)]
    c_space = np.array([(c_image[i] != 0) for i in range(4)])

    return c_space

# template of environment
class OurAwesomeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, seed=42, start=(10, 11, 3), goal=(25, 22, 2), eps=0.2):  # add other parameters
        
        np.random.seed(seed) # to "fix" randomness

        self.init_state = start # start and goal states
        self.goal_state = goal
        self.state = self.init_state

        self.g_map = generate_map() # generate map with obstacles
        self.map_sizes = self.g_map.shape 
        self.agent_image = agent_image()

        self.collission_map = get_collision_map(self.g_map, self.agent_image)

        self.observation_space = (
            spaces.Discrete(self.map_sizes[0] * self.map_sizes[1] * 4)
        )  # https://gymnasium.farama.org/api/spaces/
        # self.action_space = spaces.Discrete(8)

        self.action_space = spaces.Discrete(6)

        # enumerate actions
        self.index2action = [(1, 0, 0), 
                            (0, 1, 0), 
                            (-1, 0, 0), 
                            (0, -1, 0), 
                            (0, 0, 1), 
                            (0, 0, -1)] 

        self.counter = 0 # to truncate an episode
        self.max_steps = 100
        self.eps=eps

    def reset(self, seed=0) -> int:  # reset environment
        super().reset(seed=seed)

        self.counter = 0 # reset counter and state
        self.state = self.init_state[:]

        return (self._vect2ind(self.state), None)

    def _ind2vect(self):
        pass

    def _vect2ind(self, state): # convert vector to index
        y, x, angle = state
        h, w = self.map_sizes
        return (angle * h * w + y * w + x)

    def sample_action(self, action): # 1 - eps - probability of chosen action; eps - P(other_action) 
        probs = np.ones(self.action_space.n) * self.eps / (self.action_space.n - 1)
        probs[action] = 1 - self.eps

        return np.random.choice(list(range(len(self.index2action))), p=probs)


    def is_collisions(self, state): # check collisions
        y, x, angle = state
        
        # if abs(self.map_sizes[0] - y) > self.map_sizes[0] // 2 or abs(self.map_sizes[1] - x) > self.map_sizes[1] // 2:
        #     return True
        if x >= self.collission_map.shape[2] or y >= self.collission_map.shape[1] or x < 0 or y < 0:
            return True
        return self.collission_map[angle, y, x]

    def step(self, action: int) -> (
        int,
        float,
        bool,
        bool,
        Optional[str],
    ):  # (new_state, reward, terminated, truncated, info)

        sample_action = self.index2action[self.sample_action(action)] # sample action

        ay, ax, aangle = sample_action 
        y, x, angle = self.state

        self.state = (y+ay, x+ax, (angle+aangle)%4)
        ind4state = self._vect2ind(self.state)

        self.counter += 1

        if self.state == self.goal_state:
            return (ind4state, 1.0, True, False, None) # desired goal
        elif self.is_collisions(self.state):
            return (ind4state, -1.0, True, False, None) # collided
        elif self.counter > self.max_steps:
            return (ind4state, 0.0, False, True, None) # too long episode
        else:
            return (ind4state, 0.0, False, False, None) # simple transition
        


    def close(self):
        pass

    def render(self):
        """
        @param img: original image in 2d
        @param obj: is the 3d array of different configurations
        @param state: is the curent pose (x, y, orientation) of the object

        @return: the merged image
        """
        y, x, angle = self.state

        dims = self.agent_image[0, :, :].shape
        dim_y = int((dims[0] - 1) / 2)
        dim_x = int((dims[1] - 1) / 2)
        merged_img = np.copy(self.g_map)
        merged_img[y - dim_y:y + dim_y + 1, x - dim_x:x + dim_x + 1] += self.agent_image[angle, :, :] * 0.5
        return cv2.resize(((merged_img > 0) * 255.0).astype(np.uint8), (400, 400), cv2.INTER_LINEAR)