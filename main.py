import cv2
import numpy as np
from ultralytics import YOLO
from transformers import VivitImageProcessor, VivitForVideoClassification
import torch
def main():
    box_model = YOLO("yolo11n.pt")
    pose_model = YOLO("yolo11n-pose.pt")
    #seg_model = YOLO("yolo11n-seg.pt")

    cap = cv2.VideoCapture("videos/video.mp4")
    frame_count = 0
    while cap.isOpened():
        if frame_count % 4 == 0:
            ret, frame = cap.read()
            if not ret:
                break
            board_results = box_model.predict(frame, conf=0.5, classes=[36])[0]
            pose_results = pose_model.predict(frame, conf=0.5)[0]

            # FOR TESTING _____________________________________
            annotated_frame = board_results.plot()
            annotated_frame2 = pose_results.plot()
            combined_frame = cv2.addWeighted(annotated_frame, 0.5, annotated_frame2, 0.5, 0 )
            cv2.imshow("YOLO Detection", combined_frame)
            # __________________________________________________

            # this will for boxing to give to the Temporal CNN and ViViT
            img_height, img_width = frame.shape[:2]

            if len(board_results.boxes ) > 0 and len(pose_results.boxes) > 0:
                #get the multiplier of the board results multiply it with the height and width print it
                board_x, board_y, board_width, board_height = board_results.boxes[0].xywhn.flatten()
                board_x, board_y, board_width, board_height = board_x * img_width, board_y * img_height, board_width * img_width, board_height * img_height
                print(f"board: {board_x}, {board_y}, {board_width}, {board_height}")
                # same thing but with the pose results
                human_x, human_y, human_width, human_height = pose_results.boxes[0].xywhn.flatten()
                human_x, human_y, human_width, human_height = human_x * img_width, human_y * img_height, human_width * img_width, human_height * img_height
                print(f"human: {human_x}, {human_y}, {human_width}, {human_height}")
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()