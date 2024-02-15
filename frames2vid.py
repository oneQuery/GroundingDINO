from natsort import natsorted
from glob import glob
import cv2
import os
import numpy as np
from tqdm import tqdm


def main():
    FRAMES_DIR = 'outputs/inference/깨끗한나라_영상분석_frames/42.지게차협착(오탐).MP4'
    OUT_VID_PATH = 'outputs/inference/깨끗한나라_영상분석_vid/42.지게차협착(오탐).MP4'

    frame_paths = natsorted(glob(f'{FRAMES_DIR}/*.jpg'))

    # Use Python's built-in open function to read the image file in binary mode
    with open(frame_paths[0], 'rb') as f:
        image_data = f.read()
    # Convert the image data to a NumPy array, and then decode it into an image
    frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(OUT_VID_PATH, fourcc, 30, (width, height))
    os.makedirs(os.path.dirname(OUT_VID_PATH), exist_ok=True)

    for frame_path in tqdm(frame_paths):
        with open(frame_path, 'rb') as f:
            image_data = f.read()
        frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        video.write(frame)

    video.release()


if __name__ == '__main__':
    main()
