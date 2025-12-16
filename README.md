# Skateboard_Trick_Classifier
An end-to-end computer vision pipeline for classification of skateboard tricks using OpenCV and YOLOv8. Detecting the skater and skateboard extracting keypoints, and applies temporal model to classify the trick

---

## Current goals:
  -Reliable classification of tricks listed below:
   - Ollie
   - Pop shuvit
   - Kickflip
  -Build an extensible detection --> pose --> classification pipeline
  
---

## Future Goals:
  -Expand to more tricks that are more complex to classify (e.g., treflips)
  -Contanizerize using Docker
  -Deploy using AWS
  
---

## Dataset
Training data consists of short video clips extracted from publicly available skateboarding videos(mostly from Youtube) The dataset includes ollie, pop shuvit, and kickflip examples(for now).

The video clips are not distributed by me due to copyght considerations and ethical use. I'm developing this project with respect to privacy and ethical data use. This model solely classifies the trick rather than recognizing people.

Any sample media that is included, if applicable, will be recorded by the author of the author.
