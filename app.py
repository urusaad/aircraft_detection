import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO("best_new.pt")

# Function to detect aircraft in video
def detect_aircrafts(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (YOLOv8 expects RGB by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv8 model on the frame
        results = model(frame_rgb)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()  # This handles all visualization
        
        # Display the annotated frame
        stframe.image(annotated_frame, channels="RGB")

    cap.release()

# Streamlit UI
st.title("Aircraft Detection in Video using YOLOv8")
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    detect_aircrafts(tfile.name)
