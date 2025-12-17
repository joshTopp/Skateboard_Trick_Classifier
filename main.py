import cv2
import numpy as np
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from ultralytics import YOLO
import mediapipe as mp
from transformers import VivitImageProcessor, VivitForVideoClassification
import torch
def main():

    model = YOLO("yolo11n.pt")
    model2 = YOLO("yolo11n-pose.pt")
    seg_model = YOLO("yolo11n-seg.pt")

    cap = cv2.VideoCapture("videos/video.mp4")
    frame_count = 0
    while cap.isOpened():
        print("hello")
        if frame_count % 2 == 0:
            ret, frame = cap.read()
            if not ret:
                break
            seg_results = seg_model(frame, conf=0.5)
            pose_results = model2(frame, conf=0.5)
            #draw_skateboard(frame, results)
            #draw_pose(frame, pose_results)
            annotated_frame = seg_results[0].plot()
            annotated_frame2 = pose_results[0].plot()
            combined_frame = cv2.addWeighted(annotated_frame, 0.5, annotated_frame2, 0.5, 0 )
            cv2.imshow("YOLO Detection", combined_frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()