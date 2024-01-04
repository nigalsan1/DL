

import utils.create_video as V

test_state = states[2]
test = test_state.reshape((-1,4,84,84))
V.create_gif([frame[0].cpu().numpy() for frame in test], "modelInput", fps=10)



V.create_gif([frame[0] for frame in obss[done_idxs[2]:done_idxs[3]]], "buf45_traj3", fps=30)
