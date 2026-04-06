# Edge-AI Framework for Real-Time Surface Anomaly Detection

## 📌 Overview
This project presents a real-time surface anomaly detection system using Edge AI and deep learning. The system detects cracks and surface defects from live video streams using a lightweight YOLOv8 model.

A distributed architecture is used where a Raspberry Pi captures and streams video, while a PC performs deep learning inference. This ensures real-time performance with low latency and high accuracy.

---

## 🚀 Key Features
- Real-time crack detection using YOLOv8n
- Edge-AI based distributed architecture (Raspberry Pi + PC)
- Wireless video streaming using Flask (MJPEG)
- High accuracy with optimized dataset and training
- Portable and cost-effective system
- No dependency on cloud or internet

---

## 🧠 System Architecture
The system is divided into three main components:

1. **Video Capture (Edge Device)**
   - Raspberry Pi + Pi Camera Module
   - Captures real-time video frames
   - Streams frames over local network

2. **Processing & Detection**
   - PC / Laptop
   - YOLOv8n model for anomaly detection
   - ONNX Runtime for optimized inference

3. **Output & Visualization**
   - Displays bounding boxes on detected cracks
   - Real-time monitoring via browser or local display

---

## 🛠️ Tech Stack
- Python
- OpenCV
- Ultralytics YOLOv8
- Flask (MJPEG Streaming)
- ONNX Runtime
- NumPy

---
