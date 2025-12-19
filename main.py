import os
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import VivitImageProcessor, VivitForVideoClassification
import torch

def main():
    box_model = YOLO("yolo11n.pt")
    pose_model = YOLO("yolo11n-pose.pt")
    #seg_model = YOLO("yolo11n-seg.pt")
    video_count = 1
    dict= {}
    for video_filename in os.listdir("test"):
        print(video_filename)
        dict[video_count]= {}
        dict[video_count]["skateboard"] = []
        #dict[video_count]["human"] = []
        dict[video_count]["label"] = None
        skateboard_frames = []
        #human_frames = []
        for video in os.listdir("test/"+video_filename):
            cap = cv2.VideoCapture("test/"+video_filename+"/"+video)
            frame_count = 0
            while cap.isOpened():

                if frame_count % 4 == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    board_results = box_model.predict(frame, conf=0.3, classes=[36])[0]
                    pose_results = pose_model.predict(frame, conf=0.5)[0]

                    # FOR TESTING _____________________________________
                    annotated_frame = board_results.plot()
                    annotated_frame2 = pose_results.plot()
                    combined_frame = cv2.addWeighted(annotated_frame, 0.5, annotated_frame2, 0.5, 0 )
                    cv2.imshow("YOLO Detection", combined_frame)
                    #cv2.imshow("Frame", annotated_frame)
                    #cv2.imshow("Frame", annotated_frame2)
                    # __________________________________________________

                    # # this will for boxing to give to the Temporal CNN and ViViT
                    img_height, img_width = frame.shape[:2]

                    if len(board_results.boxes ) > 0:
                        #get the multiplier of the board results multiply it with the height and width print it
                        board_x, board_y, board_x2, board_y2 = board_results.boxes.xyxy[0]
                        board_x = max(0, int(board_x.item()))
                        board_y = max(0, int(board_y.item()))
                        board_x2 = min(img_width, int(board_x2.item()))
                        board_y2 = min(img_height, int(board_y2.item()))
                        print(f"board: {board_x}, {board_y}, {board_x2}, {board_y2}")
                        #cv2.imshow("Frame", frame[board_y:board_y2, board_x:board_x2])
                        skateboard_frames.append(frame[board_y:board_y2, board_x:board_x2])

                    #if len(pose_results.boxes ) > 0:
                        # #same thing but with the pose results
                        # human_x, human_y, human_x2, human_y2 = pose_results.boxes.xyxy[0]
                        # human_x = max(0, int(human_x.item()))
                        # human_y = max(0, int(human_y.item()))
                        # human_x2 = min(img_width, int(human_x2.item()))
                        # human_y2 = min(img_height, int(human_y2.item()))
                        # print(f"human: {human_x}, {human_y}, {human_x2}, {human_y2}")
                        # cv2.imshow("Frame", annotated_frame2[human_y:human_y2, human_x:human_x2])
                        # human_frames.append(frame[human_y:human_y2, human_x:human_x2])

                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            dict[video_count]["skateboard"] = skateboard_frames
            #dict[video_count]["human"] = human_frames
            dict[video_count]["label"] = video_filename
            video_count+=1
if __name__ == "__main__":
    main()