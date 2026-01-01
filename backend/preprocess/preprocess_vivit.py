import cv2
import numpy as np
import torch
from PIL import Image


def prep_to_train_vivit(dict_frames, indices):
    label_dict = {"kickflip": 0, "ollie": 1, "popshuv": 2}
    labels = []
    clips = []
    for video_index in indices:
        label = label_dict[dict_frames[video_index]["label"]]
        board_list = dict_frames[video_index]["skateboard"]
        for begin_index in range(0, len(board_list) - 33, 24):
            clip = board_list[begin_index:begin_index + 32]
            if len(clip) == 32:
                clips.append(clip)
                labels.append(label)

    return clips, torch.tensor(labels)



def get_boarding_boxes_vivit(results, img_height, img_length, frame):
    if len(results.boxes) == 0:
        return Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))

    box_x, box_y, box_x2, box_y2 = results.boxes.xyxy[0]
    box_x = max(0, int(box_x.item()))
    box_y = max(0, int(box_y.item()))
    box_x2 = min(img_length, int(box_x2.item()))
    box_y2 = min(img_height, int(box_y2.item()))

    if box_x2 <= box_x or box_y2 <= box_y:
        return Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
    cropped = frame[box_y:box_y2, box_x:box_x2]

    if cropped.size == 0:
        return Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))

    pil_frame = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return pil_frame
