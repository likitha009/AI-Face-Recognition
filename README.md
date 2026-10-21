# AI Face Recognition Module

## Project Description

This project is an AI-based employee face recognition module developed using Python, OpenCV, NumPy and InsightFace.

The system detects a face from an image URL, generates a face embedding and compares it with stored employee face embeddings.

## Features

- Face detection
- Face embedding generation
- Multi-employee face recognition
- Image URL based input
- Similarity-based face matching
- Unknown person detection

## Technologies Used

- Python
- OpenCV
- InsightFace
- NumPy
- Requests
- ONNX Runtime

## Project Structure

```text
Face_recogination_project/
│
├── app.py
├── src/
│   └── create_embeddings.py
├── models/
│   └── face_database.pkl
├── requirements.txt
├── README.md
└── .gitignore