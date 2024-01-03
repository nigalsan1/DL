import gymnasium as gym
# import gym
from mingpt.model_atari import GPT, GPTConfig
from mingpt.utils import sample
import numpy as np
import cv2
import torch
import ale_py.roms
# ale_py.roms.Pong = "D:/Uni/Deep_Learning/Legacy_Roms/pong.bin"
# ale_py.roms.Pong = r"C:\Users\thiem\Downloads\pong.bin"
ale_py.roms.Pong = r"C:\Users\thiem\Downloads\atari-py-0.2.5\atari-py-0.2.5\atari_py\atari_roms\pong.bin"



env = gym.make("ALE/Pong-v5", render_mode="human", obs_type="grayscale")
observation, info = env.reset()

# Load Model
mconf = GPTConfig(6, 150, n_layer=6, n_head=8, n_embd=128, model_type="reward_conditioned", max_timestep=2369)
model = GPT(mconf)
# model.load_state_dict(torch.load('./checkpoints/model.pth'))
model.load_state_dict(torch.load('./checkpoints/model.pth'))

action = None
states = np.zeros((1, 50, 4*84*84))
actions = np.zeros((1, 50, 1))
stacked_frames = np.empty((4, 84, 84))
rtgs = np.ones((1,50,1))
# timesteps = torch.tensor([i for i in range(150)]).resize(1,1,1)
for i in range(1000):
    #stack first 4 frames
    if i < 4:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # append observation to stacked_frames
        stacked_frames[i] = cv2.resize(observation, (84,84))
        continue

    # Reshape stacked_frames
    reshaped_frames = stacked_frames.reshape(1, 1, -1)
    # Append reshaped_frames to states
    states = np.append(states, reshaped_frames, axis=1)
    states = states[:, 1:, :]

    # states[0,i-4] = stacked_frames.flatten()
    # states = states[1:]

    # action = env.action_space.sample()  # agent policy that uses the observation and info
    input = torch.from_numpy(states)
    input = input.unsqueeze(0)
    logits, _ = model(torch.from_numpy(states), torch.from_numpy(actions), rtgs=torch.from_numpy(rtgs), timesteps=torch.tensor([[[i]]]))
    observation, reward, terminated, truncated, info = env.step(action)

    observation = cv2.resize(observation, (84, 84)).reshape((1,84,84))
    # remove first frame and append new frame
    stacked_frames = stacked_frames[1:]
    stacked_frames = np.append(stacked_frames, observation, axis=0)



    if terminated or truncated:
        observation, info = env.reset()

env.close()
