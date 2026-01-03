from fastapi import FastAPI

app = FastAPI()

@app.get("/send_video")
def send_videos():
    return