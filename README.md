

Face Recognition Attendance System

A real-time face recognition-based attendance system built using Python, OpenCV, and DeepFace.
The system detects faces through a webcam, matches them with stored images, and records attendance with timestamps.

Overview

This project automates attendance marking by using facial recognition instead of manual entry. It leverages DeepFace's pre-trained ArcFace model for accurate face matching and OpenCV for real-time video processing.

Features
Real-time face detection using webcam
Face recognition using DeepFace (ArcFace model)
Automatic attendance logging with date and time
Duplicate attendance prevention within a session
Displays recognized names on the video feed
Handles unknown faces gracefully
Tech Stack
Python
OpenCV
DeepFace
CSV (for data storage)
OS & Datetime modules
Project Structure
face-recognition-attendance/
│
├── known_faces/            # Directory containing known face images
│   ├── person1.jpg
│   ├── person2.jpg
│
├── attendance.csv          # Attendance records (auto-generated)
├── main.py                 # Main application script
└── README.md
Installation
Clone the repository:
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
Install required dependencies:
pip install opencv-python deepface
Usage
Add images of known individuals to the known_faces/ directory
File names should represent the person's name (e.g., john.jpg)
Run the application:
python main.py
The system will:
Start the webcam
Detect and recognize faces
Mark attendance automatically
Press Q to exit the application
Attendance Format

Attendance is stored in attendance.csv in the following format:

Name,Date,Time
Jamal,2026-04-03,10:15:23
How It Works
Captures frames from webcam using OpenCV
Saves frame temporarily for processing
Uses DeepFace to compare detected face with images in known_faces/
Extracts identity from matched image
Records attendance with timestamp
Prevents duplicate entries using an in-memory set
Requirements
Python 3.7 or higher
Working webcam
Proper lighting conditions for accurate detection
