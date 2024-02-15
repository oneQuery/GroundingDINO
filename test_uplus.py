from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm


def inference(
        model,
        input_img_path,
        output_annotated_img_path,
        text_prompt,
        box_threshold,
        text_threshold,
):



    image_source, image = load_image(input_img_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    os.makedirs(os.path.dirname(output_annotated_img_path), exist_ok=True)

    # Encode the frame
    success, encoded_image = cv2.imencode('.jpg', annotated_frame)
    if success:
        try:
            # Write the encoded image to a file
            with open(output_annotated_img_path, 'wb') as f:
                f.write(encoded_image)
            print(f'Successfully saved: {output_annotated_img_path}')
        except IOError as e:
            print(f'Error saving file {output_annotated_img_path}: {e}')
    else:
        print('Encoding failed for the annotated frame.')


def main():
    # forklift ==============================================================
    # >>>>>> chosen <<<<<<<<
    # SUB_DIRS = [
    #     '깨끗한나라_영상분석_frames/41.지게차협착(오탐).MP4',
    #     ]
    # TEXT_PROMPT = "truck . forklift . forklift with loaded on . person . bicycle ."
    # BOX_THRESHOLD = 0.42
    # TEXT_THRESHOLD = 0.42

    # >>>>>> chosen <<<<<<<<
    SUB_DIRS = [
        '깨끗한나라_영상분석_frames/42.지게차협착(오탐).MP4',
    ]
    TEXT_PROMPT = "truck . forklift . forklift with loaded on . person . bicycle ."
    BOX_THRESHOLD = 0.30
    TEXT_THRESHOLD = 0.20

    # SUB_DIRS = [
    #     '깨끗한나라_영상분석_frames/41.지게차협착(오탐).MP4',
    #     '깨끗한나라_영상분석_frames/41.지게차협착(정탐).MP4',
    #     '깨끗한나라_영상분석_frames/42.지게차협착(정탐).MP4',
    #     '깨끗한나라_영상분석_frames/43.지게차협착(오탐).MP4',
    #     '깨끗한나라_영상분석_frames/43.지게차협착(정탐).MP4',
    # ]
    # TEXT_PROMPT = "truck . forklift . forklift with loaded on . person . bicycle ."
    # BOX_THRESHOLD = 0.42
    # TEXT_THRESHOLD = 0.42

    # SOS ==============================================================
    # SUB_DIRS = [
    #     '깨끗한나라_영상분석_frames/51.SOS(정탐).MP4',
    #     '깨끗한나라_영상분석_frames/53.SOS(정탐).MP4',
    # ]
    # TEXT_PROMPT = "person repairing a forklift by raising both hands . person raising both hands sending SOS signal ."
    # BOX_THRESHOLD = 0.40
    # TEXT_THRESHOLD = 0.10

    # SUB_DIRS = [
    #     '깨끗한나라_영상분석_frames/51.SOS(오탐).MP4',
    #     '깨끗한나라_영상분석_frames/51.SOS(정탐).MP4',
    #     '깨끗한나라_영상분석_frames/52.SOS(오탐).MP4',
    #     '깨끗한나라_영상분석_frames/52.SOS(정탐).MP4',
    #     '깨끗한나라_영상분석_frames/53.SOS(오탐).MP4',
    # ]
    # TEXT_PROMPT = "person raising both hands sending SOS signal ."
    # BOX_THRESHOLD = 0.4
    # TEXT_THRESHOLD = 0.01

    # hard hat ==============================================================
    # SUB_DIRS = [
    #     '깨끗한나라_영상분석_frames/21.안전모미착용(오탐).MP4',
    #     '깨끗한나라_영상분석_frames/21.안전모미착용(정탐).MP4',
    #     '깨끗한나라_영상분석_frames/22.안전모미착용(오탐).MP4',
    #     '깨끗한나라_영상분석_frames/22.안전모미착용(정탐).MP4',
    #     '깨끗한나라_영상분석_frames/23.안전모미착용(오탐).MP4',
    #     '깨끗한나라_영상분석_frames/23.안전모미착용(정탐).MP4',
    # ]
    # TEXT_PROMPT = "person . hard hat ."
    # BOX_THRESHOLD = 0.25
    # TEXT_THRESHOLD = 0.02

    # ==============================================================
    FRAME_SKIP = 1

    model = load_model(
        "groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "weights/groundingdino_swinb_cogcoor.pth",
    )

    for sub_dir in tqdm(SUB_DIRS):

        input_images_dir = f'data/{sub_dir}'
        output_annotated_images_dir = f'outputs/inference/{sub_dir}'

        input_img_paths = natsorted(glob(f'{input_images_dir}/*.jpg'))
        output_annotated_img_paths = [f'{output_annotated_images_dir}/{os.path.basename(input_img_path)}' for input_img_path in input_img_paths]

        frame_count = 0
        for input_img_path, output_annotated_img_path in zip(input_img_paths, output_annotated_img_paths):

            if frame_count % FRAME_SKIP == 0:

                inference(
                    model=model,
                    input_img_path=input_img_path,
                    output_annotated_img_path=output_annotated_img_path,
                    text_prompt=TEXT_PROMPT,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                )

            frame_count += 1


if __name__ == '__main__':
    main()