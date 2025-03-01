from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision
import io
from PIL import Image
import os
import tempfile
import shutil
import cv2
from datetime import timedelta

app = FastAPI()

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your PyTorch model
model_path = "model/model.pth"
if os.path.exists(model_path):
    model = torch.load(model_path)
    model.eval()
    print(f"Model loaded from {model_path}")
else:
    print(f"No model found at {model_path}")
    model = None


@app.post("/predict")
async def predict(file: UploadFile):
    # Check if it's a video file
    valid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail="Only video files are supported")

    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        # Copy the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # Get video information
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        # Extract basic information
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps if fps > 0 else 0

        # Format duration as HH:MM:SS
        duration = str(timedelta(seconds=int(duration_seconds)))

        # Release resources
        cap.release()

        # Remove temp file
        os.unlink(temp_path)

        # Return video info
        return {
            "prediction": f"Video Analysis Complete",
            "info": {
                "filename": file.filename,
                "resolution": f"{frame_width}x{frame_height}",
                "fps": round(fps, 2),
                "frames": frame_count,
                "duration": duration,
                "file_size_mb": round(file.size / (1024 * 1024), 2),
            },
        }

    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
