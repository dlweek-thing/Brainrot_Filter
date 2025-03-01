from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import tempfile
import shutil
import cv2
from datetime import timedelta
from pathlib import Path
import requests

app = FastAPI()

# Create models directory if it doesn't exist
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
weights_path = models_dir / "yolov5s.pt"

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained YOLOv5 model
try:
    # Download weights file if it doesn't exist
    if not weights_path.exists():
        print("Downloading YOLOv5s weights for the first time...")
        url = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
        response = requests.get(url, stream=True)
        with open(weights_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Weights saved to {weights_path}")

    # Load YOLOv5 model with local weights
    print(f"Loading model with weights from: {weights_path}")
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights_path))
    model.eval()
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
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

        # Sample frames for YOLO processing
        object_detections = {}
        frame_skip = max(1, int(fps))  # Process 1 frame per second
        frame_count = 0

        if model is not None:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every Nth frame
                if frame_count % frame_skip == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Run YOLO detection
                    results = model(rgb_frame)

                    # Extract detection results
                    detections = results.pandas().xyxy[0]
                    for _, detection in detections.iterrows():
                        class_name = detection["name"]
                        confidence = detection["confidence"]

                        if confidence > 0.5:  # Only count high confidence detections
                            if class_name in object_detections:
                                object_detections[class_name] += 1
                            else:
                                object_detections[class_name] = 1

                frame_count += 1

                # Limit to processing first 30 seconds for quick results
                if frame_count >= fps * 30:
                    break

            # Sort detections by count
            sorted_detections = sorted(
                object_detections.items(), key=lambda x: x[1], reverse=True
            )
            detection_summary = ", ".join(
                [f"{name}: {count}" for name, count in sorted_detections[:5]]
            )
        else:
            detection_summary = "YOLOv5 model not loaded, only providing video info"

        # Release resources
        cap.release()

        # Remove temp file
        os.unlink(temp_path)

        # Return video info and object detection results
        return {
            "prediction": (
                f"Objects detected: {detection_summary}"
                if object_detections
                else "No objects detected"
            ),
            "info": {
                "filename": file.filename,
                "resolution": f"{frame_width}x{frame_height}",
                "fps": round(fps, 2),
                "frames": frame_count,
                "duration": duration,
                "file_size_mb": round(file.size / (1024 * 1024), 2),
                "detections": object_detections,
            },
        }

    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
