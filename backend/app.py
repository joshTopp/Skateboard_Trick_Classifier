import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
app = FastAPI()

@app.get("/send_video")
async def send_videos(file: UploadFile = File(...)):
    suffix = "." + file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp = tmp.name


