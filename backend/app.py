import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
from main import get_boxes
from preprocess.preprocess_vivit import prep_to_train_vivit
from train.trainViViT import TrainViViT
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import time
from contextlib import asynccontextmanager

model_instance = None
#so I can only load the model once
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    model_instance = TrainViViT()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/send_video")
async def send_videos(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        return JSONResponse({"error": "File type not supported"}, status_code=400)

    # make sure it is a set of video files
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.mp4', '.avi', ".mov", ".mkv", ".webm", ".m4v"]:
        return JSONResponse({"error": "File type not supported"}, status_code=400)

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # not trying to have 1 minute and up videos in a classifier no need for it to detect just one trick
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = (frame_count / fps) if fps > 0 else 0
        cap.release()

        if duration > 10.0:
            return JSONResponse({"error": f"Video is too long ({duration:.1f}) Max length 10 secs. Try trimming it down"}, status_code=400)

        start_time = time.perf_counter()

        dict_frames = get_boxes(path, False, None)
        eval_clip, _ = prep_to_train_vivit(dict_frames, None)
        prediction_id, scores = model_instance.eval(dict_frames)

        end_time = time.perf_counter()

        class_index = {0: "kickflip", 1: "ollie", 2: "pop shuv"}
        predicted = class_index.get(prediction_id, "None")

        # later on usage for sql logging
        return {"filename": file.filename,
                "predicted": prediction_id,
                "prediction_label": predicted,
                "prediction_score": scores,
                "duration_seconds": duration,
                "inference_time": round(end_time - start_time, 4),
                "message": "Video processed successfully"}
    finally:
        if os.path.exists(path):
            os.remove(path)
