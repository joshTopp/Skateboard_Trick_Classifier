import cv2
import numpy as np
import torch


def get_boarding_boxes_cnn(results, img_height, img_length, frame):
    box_x, box_y, box_x2, box_y2 = results.boxes.xyxy[0]
    box_x = max(0, int(box_x.item()))
    box_y = max(0, int(box_y.item()))
    box_x2 = min(img_length, int(box_x2.item()))
    box_y2 = min(img_height, int(box_y2.item()))
    #  (f"Boarding box: {box_x}, {box_y}, {box_x2}, {box_y2}")
    # cv2.imshow("Frame", frame[box_y:box_y2, box_x:box_x2])
    return resize_frame(frame[box_y:box_y2, box_x:box_x2], box_x, box_y, box_x2, box_y2)

# resizes by 224x224 and adds padding so it doesn't make the board seems expanded or shrunken
def resize_frame(box_frame, box_x, box_y, box_x2, box_y2, dimensions=224):
    width = box_x2 - box_x
    height = box_y2 - box_y

    if width <= 0 or height <= 0 or box_frame.size == 0:
        return np.zeros((dimensions, dimensions, 3), dtype=np.uint8)
    # im trying to get the image to be 224x224 universally for every frame might change for human resize though
    if width > height:
        multiplier = dimensions / width
    else:
        multiplier = dimensions / height

    width = int(width * multiplier)
    height = int(height * multiplier)
    top = (dimensions - height) // 2
    bottom = top
    left = (dimensions - width) // 2
    right = left
    if top + bottom + height != dimensions:
        bottom += 1
    if left + right + width != dimensions:
        right += 1
    resized_frame = cv2.resize(box_frame, (width, height))
    resized_frame = cv2.copyMakeBorder(resized_frame, top=top, bottom=bottom,
                                       right=right, left=left, value=(0, 0, 0),
                                       borderType=cv2.BORDER_CONSTANT)
    return resized_frame


def prep_to_train_cnn(dict_frames, indices):
    label_dict = {"kickflip": 0, "ollie": 1, "popshuv": 2}
    labels = []
    clips = []
    for video_index in indices:
        label = label_dict[dict_frames[video_index]["label"]]
        board_list = dict_frames[video_index]["skateboard"]
        # human_list = dict_frames[video_index]["human"]
        for begin_index in range(0, len(board_list) - 33, 24):
            board_frames = board_list[begin_index:begin_index + 32]
            # human_frames = human_list[begin_index:begin_index + 32]
            send_frames = []
            for frame in board_frames:
                frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
                send_frames.append(frame_tensor)

            list_clip = torch.stack(send_frames)
            clips.append(list_clip)
            labels.append(label)

    return clips, labels
