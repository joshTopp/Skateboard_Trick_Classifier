import cv2
import numpy as np
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture("/videos/video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
