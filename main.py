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
                    board_results = box_model.predict(frame, conf=0.3, classes=[36], max_det=1)[0]
                    pose_results = pose_model.predict(frame, conf=0.5, classes=[0], max_det=1)[0]

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
                        revised_frame = get_boarding_boxes(board_results, img_width, img_height, frame)
                        #cv2.imshow("Frame", revised_frame)
                        skateboard_frames.append(revised_frame)

                    #if len(pose_results.boxes ) > 0:
                        # #same thing but with the pose results
                        #revised_frame = get_boarding_boxes(pose_results, img_width, img_height, frame)
                        #human_frames.append(revised_frame)

                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            dict[video_count]["skateboard"] = skateboard_frames
            #dict[video_count]["human"] = human_frames
            dict[video_count]["label"] = video_filename
            video_count+=1


def get_boarding_boxes(results, img_width, img_height, frame):
    box_x, box_y, box_x2, box_y2 = results.boxes.xyxy[0]
    box_x = max(0, int(box_x.item()))
    box_y = max(0, int(box_y.item()))
    box_x2 = min(img_width, int(box_x2.item()))
    box_y2 = min(img_height, int(box_y2.item()))
    print(f"Boarding box: {box_x}, {box_y}, {box_x2}, {box_y2}")
    # cv2.imshow("Frame", frame[board_y:board_y2, board_x:board_x2])
    return resize_frame(frame[box_y:box_y2, box_x:box_x2], box_x, box_y, box_x2, box_y2)


#resizes by 224x224 and adds padding so it doesn't make the board seems expanded or shrunken
def resize_frame(frame, box_x, box_y, box_x2, box_y2):
    length = box_x2 - box_x
    width = box_y2- box_y
    # im trying to get the image to be 224x224 universally for every frame might change for human change though
    if length > width:
        multiplier = 224/length
    else:
        multiplier = 224/width

    resized_frame = cv2.resize(frame, (length*multiplier, width*multiplier))
    resized_frame = cv2.copyMakeBorder(resized_frame, top=int((224-width)/2), bottom=int((224-width)/2),
                                       right=int((224-length)/2), left=int((224-length)/2), value=(0,0,0),
                                       borderType=cv2.BORDER_CONSTANT)
    return resized_frame


if __name__ == "__main__":
    main()