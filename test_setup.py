import cv2
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image
import tkinter
import sqlite3

print("--- Package Verification ---")
print(f"OpenCV Version: {cv2.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")

# Verify face_recognition execution on a dummy image
dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
locations = face_recognition.face_locations(dummy_image)
print("`face_recognition` engine running successfully!")

# Verify webcam feed access
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Webcam accessed successfully!")
    cap.release()
else:
    print("Warning: Webcam could not be accessed. Check camera permissions.")

print("\nEverything is working properly!")