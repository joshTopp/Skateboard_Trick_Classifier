import os
import re

import cv2
from ultralytics import YOLO
from train.trainViViT import TrainViViT
from train.trainCNN import TrainCNN
from sklearn.model_selection import train_test_split
from preprocess.preprocess_vivit import get_boarding_boxes_vivit, prep_to_train_vivit
from preprocess.preprocess_cnn import get_boarding_boxes_cnn, prep_to_train_cnn

video_count = 1

dict_frames = {}

box_model = YOLO("yolo11n.pt")

def main():

    # pose_model = YOLO("yolo11n-pose.pt")
    # seg_model = YOLO("yolo11n-seg.pt")
    for video_filename in sorted(os.listdir("backend/videos")):
        for video in sorted(os.listdir("backend/videos/" + video_filename)):
            get_boxes("backend/videos/"+video_filename+"/"+video, testing=True, video_label=video_filename)

    videos = list(dict_frames.keys())
    labels = [dict_frames[video]["label"] for video in videos]
    train_indices, test_indices = train_test_split(videos, test_size=0.2, stratify=labels, random_state=42)
    # train_clips, train_labels = prep_to_train_cnn(dict_frames, train_indices)
    # trainer = TrainCNN(train_clips, train_labels)
    # trainer.training(epochs=10)

    test_clips, test_labels = prep_to_train_vivit(dict_frames, test_indices)
    trainer = TrainViViT(test_clips, test_labels)
    # trainer.test(test_clips, test_labels)
    trainer.test(dict_frames, test_indices)


def get_boxes(video_file, testing=True, video_label=None):
    global video_count
    if testing:
        dict_frames[video_count] = {}
        dict_frames[video_count]["skateboard"] = []
        # dict_frames[video_count]["human"] = []
    else:
        classify_frames = []
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        board_results = box_model.predict(frame, conf=0.3, classes=[36], max_det=1)[0]
        # pose_results = pose_model.predict(frame, conf=0.5, classes=[0], max_det=1)[0]

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

        if len(board_results.boxes) > 0:
            # get the multiplier of the board results multiply it with the width and height print it
            revised_frame = get_boarding_boxes_vivit(board_results, img_height, img_length, frame)
            # cv2.imshow("Frame", revised_frame)
            if testing:
                dict_frames[video_count]["skateboard"].append(revised_frame)
            else:
                classify_frames.append(revised_frame)
            # if len(pose_results.boxes ) > 0:
            #     #same thing but with the pose results
            #     human_revised_frame = get_boarding_boxes(pose_results, img_height, img_length, frame)
            #     dict_frames[video_count]["human"].append(human_revised_frame)
            # elif human_revised_frame is not None:
            #     dict_frames[video_count]["human"].append(human_revised_frame)
            # else:
            #     blank_frame = np.zeros(shape=[224,224,3], dtype=np.uint8)
            #     dict_frames[video_count]["human"].append(blank_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if testing:
        dict_frames[video_count]["label"] = video_label
        dict_frames[video_count]["num_frames"] = len(dict_frames[video_count]["skateboard"])
        video_count += 1

    else:
        length_frames = len(classify_frames)
        test_clips, test_labels = prep_to_train_cnn()


if __name__ == "__main__":
    main()
