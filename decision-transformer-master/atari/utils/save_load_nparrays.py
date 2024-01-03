

import numpy as np

def save_np_arrays(obss, actions, returns, done_idxs, rtg, timesteps):

    # Example arrays
    obss = np.array([1, 2, 3, 4, 5])
    actions = np.array([6, 7, 8, 9, 10])
    returns = np.array([11, 12, 13, 14, 15])
    done_idxs = np.array([16, 17, 18, 19, 20])
    rtg = np.array([21, 22, 23, 24, 25])
    timesteps = np.array([26, 27, 28, 29, 30])

    # Save to .npz file
    np.savez('arrays.npz', obss=obss, actions=actions, returns=returns, done_idxs=done_idxs, rtg=rtg, timesteps=timesteps)


def load_np_arrays():
    # Load the arrays from .npz file
    with np.load('arrays.npz') as data:
        obss = data['obss']
        actions = data['actions']
        returns = data['returns']
        done_idxs = data['done_idxs']
        rtg = data['rtg']
        timesteps = data['timesteps']
    return obss, actions, returns, done_idxs, rtg, timesteps

if __name__ == "__main__":
    test = 0