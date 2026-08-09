# Face Mask Compliance Screening System

A computer vision-based application for automated face-mask compliance screening.

The system captures an image through the user's camera, detects faces using OpenCV's deep learning-based face detector, and classifies each detected face as **Mask** or **No Mask** using a trained deep learning model.

The application provides a clear screening result with detection confidence and an overall compliance decision.

## Live 
(https://face-mask-detection-yqum2z76nympbrbx3peqql.streamlit.app/)

## Overview

Manual checking of face-mask compliance at controlled entry points can be time-consuming and inconsistent.

This project provides a computer vision-based screening approach where a captured image is automatically analyzed to:

- Detect one or more faces
- Determine mask status for each detected face
- Display the prediction confidence
- Count compliant and non-compliant faces
- Provide an overall screening decision

The system can be used as a screening-support solution in environments where face-mask requirements are applicable.

## Key Features

- Camera-based image capture
- Automatic face detection
- Face-mask classification
- Multiple-face detection in a single image
- Prediction confidence scores
- Compliant / Non-Compliant classification
- Screening summary
- Professional Streamlit dashboard
- Cloud deployment using Streamlit Community Cloud

## Application Workflow

```text
Camera Capture
      ↓
Image Processing
      ↓
Face Detection
      ↓
Face Region Extraction
      ↓
Mask Classification
      ↓
Confidence Calculation
      ↓
Compliance Assessment
      ↓
Screening Result
