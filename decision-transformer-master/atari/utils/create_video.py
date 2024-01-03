import imageio
import numpy as np


def create_video(frames):
    # Create a GIF

    # frames = np.random.randint(0, 256, (50, 84, 84), dtype=np.uint8)  # Example data
    with imageio.get_writer('my_video.mp4', fps=10) as writer:
        for frame in frames:
            writer.append_data(frame)



if __name__ == "__main__":
    create_video(np.array(frame in obss))