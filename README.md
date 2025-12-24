# Skateboard Trick Classifier
An end-to-end computer vision pipeline for classification of skateboard tricks using OpenCV and YOLO11. Using YOLO pose estimation for the skater and YOLO boxing for the skateboard extracting keypoints (I used YOLO because MediaPipe doesn't do well in this situation as I tested and OpenPose seems a little outdated. I was planning using HRNet but there is no point to complicate it as YOLO is really accurate) I plan on starting out using a temporal CNN, but later I will compare with an optimized frozen param ViVit Hugging Face model.

---

## Current goals:
 - Reliable classification of tricks listed below: 
   - Ollie
   - Pop shuvit
   - Kickflip
 
 - Implement a simple Fast API frontend to allow you to submit a trick video
 - Video → Detection → Pose Estimation → Temporal Modeling → Classify → Deploy

---

## Future Goals:
  - Expand to more tricks that are more complex to classify (e.g., treflips, heelflips, laserflip, 3shuv, body varials, etc.)

  - Containerize using Docker

  - A simple deploy using AWS

-----

## Dataset
Training data consists of short video clips extracted from publicly available skateboarding videos (mostly from Youtube). The dataset includes ollie, pop shuvit, and kickflip examples.

The video clips are not distributed by me due to copyright considerations and ethical use. I'm developing this project with respect to privacy and ethical data use. This model solely classifies the trick rather than recognizing people.

Any sample media that is included, if applicable, will be recorded by the author.
