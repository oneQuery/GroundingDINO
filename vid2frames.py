import os
import cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm


def vid2frames(video_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print(f'No more frames to read at {count}')
            break

        # if count % 30 == 0:
        output_filepath = rf'{output_dir}\frame_{count}.jpg'
        # Encode the frame
        success, encoded_image = cv2.imencode('.jpg', frame)
        if success:
            # Write the encoded image to a file
            with open(output_filepath, 'wb') as f:
                f.write(encoded_image)
            print(f'frame_{count}.jpg saved')
        else:
            print(f'Encoding failed for frame {count}')

        count += 1

    cap.release()


def main():
    VIDEO_DIR = r'data\깨끗한나라_영상분석'

    video_paths = natsorted(glob(f'{VIDEO_DIR}/*.MP4'))
    output_dir = [f'outputs/{os.path.basename(video_path)}' for video_path in video_paths]

    for video_path, output_dir in tqdm(zip(video_paths, output_dir)):
        print(f'Processing {video_path} to {output_dir}')
        vid2frames(video_path, output_dir)


if __name__ == '__main__':
    main()
