from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import tempfile
import shutil
import cv2
from pathlib import Path
from inference_sdk import InferenceHTTPClient
import pandas as pd
import asyncio
import json
import uuid
from typing import Dict, Any

app = FastAPI()

# Enable CORS for frontend with proper settings for SSE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for progress tracking
progress_store: Dict[str, Dict[str, Any]] = {}


@app.post("/predict")
async def predict(file: UploadFile):
    # Generate unique ID for this prediction
    prediction_id = str(uuid.uuid4())

    # Initialize progress in store
    progress_store[prediction_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting video processing",
        "result": None,
    }

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

        # Release resources
        cap.release()

        # Start skibidi score processing in background
        asyncio.create_task(
            process_video_async(
                temp_path,
                prediction_id,
                file.filename,
                frame_width,
                frame_height,
                fps,
                frame_count,
                duration_seconds,
                file.size,
            )
        )

        # Return prediction ID for tracking progress
        return {"prediction_id": prediction_id, "message": "Video processing started"}

    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        progress_store[prediction_id] = {
            "status": "error",
            "message": f"Error processing video: {str(e)}",
        }
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


async def process_video_async(
    video_path,
    prediction_id,
    filename,
    frame_width,
    frame_height,
    fps,
    frame_count,
    duration_seconds,
    file_size,
):
    try:
        # Calculate skibidi score with progress updates
        skibidi_score = await predict_skibidi_score_with_progress(
            video_path, prediction_id
        )

        # Update result
        progress_store[prediction_id] = {
            "status": "complete",
            "progress": 100,
            "message": "Processing complete",
            "result": {
                "prediction": f"Skibidi Score: {skibidi_score:.2f}",
                "info": {
                    "filename": filename,
                    "resolution": f"{frame_width}x{frame_height}",
                    "fps": round(fps, 2),
                    "frames": frame_count,
                    "duration": duration_seconds,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "skibidi_score": skibidi_score,
                },
            },
        }
    except Exception as e:
        progress_store[prediction_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Error: {str(e)}",
        }
    finally:
        # Clean up temp file
        if os.path.exists(video_path):
            os.unlink(video_path)


@app.get("/progress/{prediction_id}")
async def progress(prediction_id: str):
    # Server-sent events endpoint
    async def event_generator():
        while True:
            if prediction_id not in progress_store:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            progress_data = progress_store[prediction_id]
            yield f"data: {json.dumps(progress_data)}\n\n"

            # If processing is complete or error occurred, stop sending updates
            if progress_data["status"] in ["complete", "error"]:
                # Clean up after some time
                await asyncio.sleep(60)  # Keep result for 60 seconds
                if prediction_id in progress_store:
                    del progress_store[prediction_id]
                break

            await asyncio.sleep(0.5)  # Update every 0.5 seconds

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def predict_skibidi_score_with_progress(video_path, prediction_id):
    # Create a temporary directory for frames
    output_dir = tempfile.mkdtemp(prefix="tempstor_images_")

    # Update progress
    progress_store[prediction_id] = {
        "status": "processing",
        "progress": 5,
        "message": "Starting video processing",
    }

    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not video.isOpened():
            raise Exception(f"Error opening video file")

        # Get total frame count for progress tracking
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Update progress
        progress_store[prediction_id] = {
            "status": "processing",
            "progress": 10,
            "message": f"Extracting frames from video",
        }

        # Process at most 300 frames to avoid excessive processing time
        max_frames_to_process = min(total_frames, 300)
        frame_step = max(1, total_frames // max_frames_to_process)

        # Read and save each frame
        frame_count = 0
        frames_saved = 0

        while True:
            ret, frame = video.read()

            # Break the loop if the end of the video is reached
            if not ret:
                break

            # Only save every Nth frame to reduce processing time
            if frame_count % frame_step == 0:
                # Save the frame as an image
                output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(output_path, frame)
                frames_saved += 1

                # Update progress (10-40%)
                if frame_count % (frame_step * 10) == 0:
                    progress_pct = 10 + min(30, (frame_count / total_frames) * 30)
                    progress_store[prediction_id] = {
                        "status": "processing",
                        "progress": round(progress_pct),
                        "message": f"Extracting frames: {frame_count}/{total_frames}",
                    }

            frame_count += 1

            # Allow other tasks to run
            if frame_count % 20 == 0:
                await asyncio.sleep(0)

        # Release the video capture object
        video.release()

        # Update progress
        progress_store[prediction_id] = {
            "status": "processing",
            "progress": 40,
            "message": f"Frame extraction complete. Starting inference...",
        }

        # Initialize inference client
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com", api_key="r2EJ6g6arI8b3S7jHK5W"
        )

        skibidi_score = pd.DataFrame(
            columns=["has_skib", "w", "h", "total_w", "total_h"]
        )

        frames_to_process = os.listdir(output_dir)
        frames_with_predictions = 0

        for i, frame in enumerate(frames_to_process):
            frame_path = os.path.join(output_dir, frame)

            # Call the inference API
            result = CLIENT.infer(frame_path, model_id="skibidi-mmt7z/5")

            # Update progress (40-90%)
            if i % 5 == 0 or i == len(frames_to_process) - 1:
                progress_pct = 40 + min(50, ((i + 1) / len(frames_to_process)) * 50)
                progress_store[prediction_id] = {
                    "status": "processing",
                    "progress": round(progress_pct),
                    "message": f"Running inference: {i+1}/{len(frames_to_process)}",
                }

            # Check if there are any predictions
            if not result["predictions"]:
                continue  # Skip frames with no predictions

            frames_with_predictions += 1
            has_skib = 1 if result["predictions"][0]["confidence"] > 0.5 else 0
            width = result["predictions"][0]["x"]
            height = result["predictions"][0]["y"]
            total_w = result["image"]["width"]
            total_h = result["image"]["height"]

            skibidi_score.loc[len(skibidi_score)] = [
                has_skib,
                width,
                height,
                total_w,
                total_h,
            ]

            # Allow other tasks to run
            if i % 5 == 0:
                await asyncio.sleep(0)

        # Update progress
        progress_store[prediction_id] = {
            "status": "processing",
            "progress": 90,
            "message": "Calculating final score",
        }

        # If no frames had predictions, return 0
        if len(skibidi_score) == 0:
            return 0

        skibidi_score["frame_score"] = skibidi_score["has_skib"] * (
            (skibidi_score["w"] * skibidi_score["h"])
            / (skibidi_score["total_w"] * skibidi_score["total_h"])
        )

        final_score = skibidi_score["frame_score"].mean()

        # Update progress
        progress_store[prediction_id] = {
            "status": "processing",
            "progress": 95,
            "message": f"Score calculation complete: {final_score:.4f}",
        }

        return final_score

    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
