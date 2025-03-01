# -*- coding: utf-8 -*-
"""Deploy.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zjDEKmy057gEL_Cn6kq22rXVwIN8zS34
"""

#Installations

import cv2
import os
from inference_sdk import InferenceHTTPClient
import pandas as pd

def predict_skibidi_score(video_path):

  # Create the output directory if it doesn't exist
  output_dir = './tempstor_images'
  os.makedirs(output_dir, exist_ok=True)

  # Open the video file
  # video_path = '/content/SnapTik_App_7362440000333565189.mp4'
  video = cv2.VideoCapture(video_path)

  # Check if the video opened successfully
  if not video.isOpened():
      print(f"Error opening video file: {video_path}")
      exit()

  # Read and save each frame
  frame_count = 0
  while True:
      ret, frame = video.read()

      # Break the loop if the end of the video is reached
      if not ret:
          break

      # Save the frame as an image
      output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
      cv2.imwrite(output_path, frame)
      frame_count += 1

  # Release the video capture object
  video.release()

  # print(f"Successfully extracted {frame_count} frames to {output_dir}")
  CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="r2EJ6g6arI8b3S7jHK5W")

  skibidi_score = pd.DataFrame(columns=['has_skib','w', 'h','total_w','total_h'])

  for frame in os.listdir('/content/tempstor_images'):
    # print(frame)
    result = CLIENT.infer(f"/content/tempstor_images/{frame}", model_id="skibidi-mmt7z/5")
    # print(result)
    has_skib = 1 if result['predictions'][0]['confidence'] > 0.5 else 0
    width = result['predictions'][0]['x']
    height = result['predictions'][0]['y']
    total_w = result['image']['width']
    total_h = result['image']['height']
    skibidi_score.loc[len(skibidi_score)] = [has_skib, width, height, total_w, total_h]

  skibidi_score['frame_score']=skibidi_score['has_skib']*((skibidi_score['w']*skibidi_score['h'])/(skibidi_score['total_w']* skibidi_score['total_h']))

  return skibidi_score['frame_score'].mean()

# import the inference-sdk