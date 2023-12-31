import gymnasium as gym
from mingpt.model_atari import GPT, GPTConfig
from mingpt.utils import sample
import numpy as np
import cv2
import torch
import ale_py.roms
ale_py.roms.Pong = "D:/Uni/Deep_Learning/Legacy_Roms/pong.bin"
ale_py.roms.Tennis = "D:/Uni/Deep_Learning/Legacy_Roms/tennis.bin"
# ale_py.roms.Pong = r"C:\Users\thiem\Downloads\atari-py-0.2.5\atari-py-0.2.5\atari_py\atari_roms\pong.bin"
# ale_py.roms.Tennis = r"C:\Users\thiem\Downloads\atari-py-0.2.5\atari-py-0.2.5\atari_py\atari_roms\tennis.bin"
game = "Tennis"
random_player = True
n_steps = 100000
context_length = 50


if game == "Pong":
    vocab_size = 6
    target_return = 21
elif game == "Tennis":
    vocab_size = 18
    target_return = 1

env = gym.make(f"ALE/{game}-v5", render_mode="human", obs_type="grayscale", frameskip=1)
observation, info = env.reset()

if not random_player:
    # Load Model
    mconf = GPTConfig(vocab_size, 150, n_layer=6, n_head=8, n_embd=128, model_type="reward_conditioned", max_timestep=4731)
    model = GPT(mconf)
    model.load_state_dict(torch.load(f'./saved_checkpoints/model_{game}_0.pth'))
    model.eval()

action = None
states = np.zeros((1, context_length, 4*84*84))
actions = np.zeros((1, context_length, 1))
stacked_frames = np.empty((4, 84, 84))
rtgs = np.ones((1,context_length,1))
# timesteps = torch.tensor([i for i in range(n_steps)]).resize(1,1,1)



for i in range(n_steps):

    # Get Action every 4th frame
    if i == 0 or random_player:
        action = env.action_space.sample()
        # Append action to actions
        actions = np.append(actions, np.array([action]).reshape(1,1,1), axis=1)
        actions = actions[:, 1:, :]

    elif i % 4 == 0:
        # Reshape stacked_frames
        reshaped_frames = stacked_frames.reshape(1, 1, -1)

        # Append reshaped_frames to states
        states = np.append(states, reshaped_frames, axis=1)
        states = states[:, 1:, :]

        input = torch.from_numpy(states)
        input = input.unsqueeze(0)

        logits, _ = model(torch.from_numpy(states), torch.from_numpy(actions), rtgs=torch.from_numpy(rtgs), timesteps=torch.tensor([[[int(i/4)]]], dtype=torch.int64))
        logits = logits[:, -1, :] # pluck the logits at the final step
        action = int(logits.argmax(dim=-1))

        # Append action to actions
        actions = np.append(actions, np.array([action]).reshape(1,1,1), axis=1)
        actions = actions[:, 1:, :]



    observation, reward, terminated, truncated, info = env.step(action)
    if terminated: print("terminated")
    # append observation to stacked_frames
    stacked_frames[i%4] = cv2.resize(observation, (84,84))

env.close()
