import imageio
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fixed_replay_buffer import FixedReplayBuffer

def create_gif(frames, num, fps=10, ):
    # Create a GIF

    # frames = np.random.randint(0, 256, (50, 84, 84), dtype=np.uint8)  # Example data
    with imageio.get_writer(f'my_video_{num}.mp4', fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


if __name__ == '__main__':
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []
    transitions_per_buffer = np.zeros(50, dtype=int)
    trajectories_per_buffer = 1
    num_trajectories = 0
    buffer_num = 0
    i = transitions_per_buffer[buffer_num]
    print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
    frb = FixedReplayBuffer(
        data_dir=r"D:\Uni\Deep_Learning\DL\decision-transformer-master\atari\data\Tennis/1/replay_logs",
        replay_suffix=buffer_num,
        observation_shape=(210, 160),
        stack_size=4,
        update_horizon=1,
        gamma=0.99,
        observation_dtype=np.uint8,
        batch_size=32,
        replay_capacity=100000)
    if frb._loaded_buffers:
        done = False
        curr_num_transitions = len(obss)
        trajectories_to_load = trajectories_per_buffer
        while not done:
            states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(
                batch_size=1, indices=[i])
            states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
            obss += [states]
            actions += [ac[0]]
            stepwise_returns += [ret[0]]
            if terminal[0]:
                done_idxs += [len(obss)]
                returns += [0]
                if trajectories_to_load == 0:
                    done = True
                else:
                    trajectories_to_load -= 1
            returns[-1] += ret[0]
            i += 1
            if i >= 100000:
                print("more than 100000 transitions in this buffer %d, resetting" % buffer_num)
                obss = obss[:curr_num_transitions]
                actions = actions[:curr_num_transitions]
                stepwise_returns = stepwise_returns[:curr_num_transitions]
                done_idxs = done_idxs[:curr_num_transitions]
                returns[-1] = 0
                i = transitions_per_buffer[buffer_num]
                done = True
        num_trajectories += (trajectories_per_buffer - trajectories_to_load)
        transitions_per_buffer[buffer_num] = i
        create_gif([frame[0] for frame in obss], buffer_num, fps=30)