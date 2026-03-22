# 🚧 PotholeGuard — Real-Time Pothole Detection

> **CGIP Project** | Computer Graphics & Image Processing  
> Powered by **YOLOv12** + **OpenCV**

---

## 📸 Sample Detections

<p align="center">
  <img src="assets/pothole_detected.jpg" width="45%"/>
  <img src="assets/pothole2_detected.jpg" width="45%"/>
</p>
<p align="center">
  <img src="assets/pothole3_detected.jpg" width="45%"/>
  <img src="assets/pothole5_detected.jpg" width="45%"/>
</p>

---

## 🧠 Features

- ✅ **Real-time pothole detection** using a custom-trained YOLOv12 model
- 🎨 **Dynamic alert levels** — CAUTION / WARNING / DANGER based on rolling average
- 📹 Supports **images, videos, webcam**, and **YouTube streams**
- 🖥️ **HUD overlay** showing FPS, pothole count, and alert state
- 💾 Auto-saves annotated output to `/output`
- 📓 **Google Colab notebook** for training and inference

---

## 📁 Project Structure

```
pothole-detection/
│
├── detect_potholes.py          # Main detection script
├── requirements.txt            # Python dependencies
├── Pothole_Detection_YOLOv12_Colab.ipynb  # Training notebook
│
├── models/
│   └── best.pt                 # Trained YOLOv12 weights (download separately)
│
├── assets/                     # Sample detection images
│   ├── pothole_detected.jpg
│   ├── pothole2_detected.jpg
│   └── ...
│
└── output/                     # Auto-generated detection results
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/pothole-detection.git
cd pothole-detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model weights
Place `best.pt` inside the `models/` folder.  
> You can download it from the [Releases](../../releases) page or train your own using the Colab notebook.

---

## 🚀 Usage

### 🖼️ Image Detection
```bash
python detect_potholes.py --source path/to/image.jpg
```

### 🎬 Video Detection
```bash
python detect_potholes.py --source path/to/video.mp4
```

### 📷 Webcam (live)
```bash
python detect_potholes.py --source 0
```

### 🌐 YouTube Stream
```bash
python detect_potholes.py --source "https://www.youtube.com/watch?v=VIDEO_ID"
```

### 🛠️ All Options
```bash
python detect_potholes.py \
  --source 0 \
  --model models/best.pt \
  --conf 0.25 \
  --iou 0.4 \
  --imgsz 640 \
  --output output
```

| Argument    | Default           | Description                          |
|-------------|-------------------|--------------------------------------|
| `--source`  | `0` (webcam)      | Image / video path, 0 for webcam, or YouTube URL |
| `--model`   | `models/best.pt`  | Path to YOLO model weights           |
| `--conf`    | `0.25`            | Confidence threshold                 |
| `--iou`     | `0.4`             | IoU threshold for NMS                |
| `--imgsz`   | `640`             | Inference image size                 |
| `--output`  | `output`          | Directory for saved results          |
| `--no-save` | False             | Disable saving output                |

---

## 🚦 Alert Levels

| Level     | Avg Potholes (rolling 20 frames) | Border Color |
|-----------|----------------------------------|--------------|
| 🟢 CAUTION | ≥ 1                             | Green        |
| 🟠 WARNING | ≥ 2.5                           | Orange       |
| 🔴 DANGER  | ≥ 4                             | Red          |

---

## 📓 Google Colab Training

Open `Pothole_Detection_YOLOv12_Colab.ipynb` to:
- Train a custom YOLOv12 model on pothole datasets
- Export weights as `best.pt`
- Run inference in the cloud

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/pothole-detection/blob/main/Pothole_Detection_YOLOv12_Colab.ipynb)

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics) | Object detection model |
| [OpenCV](https://opencv.org/) | Frame processing & display |
| [PyTorch](https://pytorch.org/) | Model inference backend |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | YouTube stream extraction |

---

## 📜 License

This project is for academic/educational use under the **MIT License**.

---

## 🙋 Author

**CGIP Project** — Computer Graphics & Image Processing  
Feel free to open issues or pull requests!
