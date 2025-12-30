import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from Train.trainViViT import TrainViViT
from sklearn.model_selection import train_test_split

def main():
    box_model = YOLO("yolo11n.pt")
    pose_model = YOLO("yolo11n-pose.pt")
    #seg_model = YOLO("yolo11n-seg.pt")
    video_count = 1
    dict_frames= {}
    for video_filename in os.listdir("backend/videos"):
        for video in os.listdir("backend/videos/"+video_filename):
            print("hello")
            dict_frames[video_count] = {}
            dict_frames[video_count]["skateboard"] = []
            # dict_frames[video_count]["human"] = []
            dict_frames[video_count]["label"] = None
            skateboard_frames = []
            # human_frames = []
            cap = cv2.VideoCapture("backend/videos/"+video_filename+"/"+video)
            frame_count = 0
            while cap.isOpened():
                if frame_count % 4 == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    board_results = box_model.predict(frame, conf=0.3, classes=[36], max_det=1)[0]
                    pose_results = pose_model.predict(frame, conf=0.5, classes=[0], max_det=1)[0]

                    # FOR TESTING _____________________________________
                    # annotated_frame = board_results.plot()
                    # annotated_frame2 = pose_results.plot()
                    # combined_frame = cv2.addWeighted(annotated_frame, 0.5, annotated_frame2, 0.5, 0 )
                    # cv2.imshow("YOLO Detection", combined_frame)
                    # cv2.imshow("Frame", annotated_frame)
                    # cv2.imshow("Frame", annotated_frame2)
                    # __________________________________________________

                    # # this will for boxing to give to the Temporal CNN and ViViT
                    img_length, img_height = frame.shape[:2]

                    if len(board_results.boxes ) > 0:
                        #get the multiplier of the board results multiply it with the width and height print it
                        revised_frame = get_boarding_boxes(board_results, img_height, img_length, frame)
                        #cv2.imshow("Frame", revised_frame)
                        skateboard_frames.append(revised_frame)

                        # if len(pose_results.boxes ) > 0:
                        #     #same thing but with the pose results
                        #     human_revised_frame = get_boarding_boxes(pose_results, img_height, img_length, frame)
                        #     human_frames.append(human_revised_frame)
                        # elif human_revised_frame is not None:
                        #     human_frames.append(human_revised_frame)
                        # else:
                        #     blank_frame = np.zeros(shape=[224,224,3], dtype=np.uint8)
                        #     human_frames.append(blank_frame)

                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            dict_frames[video_count]["skateboard"] = skateboard_frames
            #dict_frames[video_count]["human"] = human_frames
            dict_frames[video_count]["label"] = video_filename
            dict_frames[video_count]["num_frames"] = len(skateboard_frames)
            video_count+=1

    videos = list(dict_frames.keys())
    labels = [dict_frames[video]["label"] for video in videos]

    train_indices, test_indices = train_test_split(videos, test_size=0.2, stratify=labels, random_state=42)
    train_clips, train_labels = send_to_neural(dict_frames, train_indices)
    trainer = TrainViViT(train_clips, train_labels)
    trainer.training(epochs=20)

    test_clips, test_labels = send_to_neural(dict_frames, test_indices)
    trainer.list_clips = test_clips
    trainer.list_labels = test_labels
    trainer.test()

def get_boarding_boxes(results, img_height, img_length, frame):
    box_x, box_y, box_x2, box_y2 = results.boxes.xyxy[0]
    box_x = max(0, int(box_x.item()))
    box_y = max(0, int(box_y.item()))
    box_x2 = min(img_length, int(box_x2.item()))
    box_y2 = min(img_height, int(box_y2.item()))
    #  (f"Boarding box: {box_x}, {box_y}, {box_x2}, {box_y2}")
    # cv2.imshow("Frame", frame[box_y:box_y2, box_x:box_x2])
    return resize_frame(frame[box_y:box_y2, box_x:box_x2], box_x, box_y, box_x2, box_y2)


#resizes by 224x224 and adds padding so it doesn't make the board seems expanded or shrunken
def resize_frame(box_frame, box_x, box_y, box_x2, box_y2, dimensions=224):
    width = box_x2 - box_x
    height = box_y2- box_y

    if width <= 0 or height <= 0 or box_frame.size == 0:
        return np.zeros((dimensions, dimensions, 3), dtype=np.uint8)
    # im trying to get the image to be 224x224 universally for every frame might change for human resize though
    if width > height:
        multiplier = dimensions/width
    else:
        multiplier = dimensions/height

    width = int(width * multiplier)
    height = int(height * multiplier)
    top = (dimensions - height) // 2
    bottom = top
    left = (dimensions - width) // 2
    right = left
    if top + bottom + height != dimensions:
        bottom+=1
    if left + right + width != dimensions:
        right+=1
    resized_frame = cv2.resize(box_frame, (width, height))
    resized_frame = cv2.copyMakeBorder(resized_frame, top=top, bottom=bottom,
                                       right=right, left=left, value=(0,0,0),
                                       borderType=cv2.BORDER_CONSTANT)
    return resized_frame

def send_to_neural(dict_frames, indices):
    label_dict = {"kickflip": 0, "ollie": 1, "popshuv": 2}
    labels = []
    clips = []
    for video_index in indices:
        label = label_dict[dict_frames[video_index]["label"]]
        frame_list = dict_frames[video_index]["skateboard"]
        for begin_index in range(0, len(frame_list) - 33 ,24):
            send_frames = frame_list[begin_index:begin_index + 32]
            list_clip = np.array(send_frames)
            list_clip = torch.tensor(list_clip)
            list_clip = list_clip.permute(0, 3, 1, 2).unsqueeze(0)
            clips.append(list_clip)
            labels.append(label)

    batch_clips = torch.cat(clips, dim=0)
    tensor_labels = torch.tensor(labels)

    return batch_clips, tensor_labels


if __name__ == "__main__":
    main()