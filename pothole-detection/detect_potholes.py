"""
Pothole Detection using YOLOv8
CGIP Project - Computer Graphics & Image Processing
Run: python detect_potholes.py --source <image/video/webcam>
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import deque
import time
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "model_path"    : "models/best.pt",
    "conf_threshold": 0.25,
    "iou_threshold" : 0.4,
    "img_size"      : 640,
    "class_names"   : ["pothole"],
    "save_output"   : True,
    "output_dir"    : "output",
    "history_size"  : 20,
}

BOX_COLOR  = (0,  60, 255)
TEXT_COLOR = (255,255,255)


def get_alert_level(avg):
    """Returns (label, color) based on rolling avg pothole count."""
    if avg >= 4:   return "DANGER",  (0,  0, 220)
    if avg >= 2.5: return "WARNING", (0, 120,255)
    if avg >= 1:   return "CAUTION", (0, 200,100)
    return None, None


def draw_boxes(frame, results, class_names):
    count = 0
    for result in results:
        if result.boxes is None: continue
        for box in result.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = class_names[cls] if cls < len(class_names) else f"cls{cls}"
            count += 1
            lbl = f"{label}"
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
            cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+6,y1),BOX_COLOR,-1)
            cv2.rectangle(frame,(x1,y1),(x2,y2),BOX_COLOR,2)
            cv2.putText(frame,lbl,(x1+3,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,TEXT_COLOR,1,cv2.LINE_AA)
    return frame, count


def draw_hud(frame, fps, count, frame_num):
    h,w = frame.shape[:2]
    ov  = frame.copy()
    cv2.rectangle(ov,(0,0),(w,38),(15,15,15),-1)
    frame = cv2.addWeighted(ov,0.7,frame,0.3,0)
    cv2.putText(frame,"CGIP | PotholeGuard [YOLOv8]",
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.62,(200,200,200),1,cv2.LINE_AA)
    cv2.putText(frame,f"FPS:{fps:.1f}  Potholes:{count}",
                (w-220,25),cv2.FONT_HERSHEY_SIMPLEX,0.55,(100,255,100),1,cv2.LINE_AA)
    return frame


def draw_alert_flash(frame, avg, frame_num):
    """Flash a colored border + alert text when potholes detected."""
    label, color = get_alert_level(avg)
    if label is None:
        return frame

    h,w = frame.shape[:2]

    # Pulsing border — thickness pulses between 4 and 10
    pulse     = abs((frame_num % 20) - 10)          # 0..10..0
    thickness = 4 + pulse // 2

    # Flash overlay every other ~10 frames
    if (frame_num // 10) % 2 == 0:
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(w,h),color,-1)
        frame = cv2.addWeighted(ov,0.10,frame,0.90,0)

    # Solid colored border
    cv2.rectangle(frame,(0,0),(w-1,h-1),color,thickness)

    # Alert text top-center
    (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.85,2)
    tx = (w-tw)//2
    ty = 75
    # dark pill behind text
    cv2.rectangle(frame,(tx-12,ty-th-8),(tx+tw+12,ty+6),(0,0,0),-1)
    cv2.rectangle(frame,(tx-12,ty-th-8),(tx+tw+12,ty+6),color,2)
    cv2.putText(frame,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.85,color,2,cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
# Main Detection Loop
# ─────────────────────────────────────────────
def get_youtube_stream_url(url):
    """Extract direct stream URL from a YouTube link using yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info["url"]
            print(f"  [yt-dlp] Stream URL extracted successfully.")
            return stream_url
    except Exception as e:
        raise RuntimeError(f"yt-dlp failed to extract stream URL: {e}")


def run_detection(source, model_path, conf, iou, img_size, save_output, output_dir):
    print(f"\n{'='*50}")
    print(f"  CGIP PotholeGuard — YOLOv8")
    print(f"  Source : {source}  |  Model : {model_path}")
    print(f"{'='*50}\n")

    model       = YOLO(model_path)
    class_names = CONFIG["class_names"]
    history     = deque(maxlen=CONFIG["history_size"])

    # ── YouTube URL handling ──────────────────────────────────
    is_youtube = isinstance(source, str) and (
        "youtube.com/watch" in source or "youtu.be/" in source or
        "youtube.com/shorts/" in source
    )
    if is_youtube:
        print("  [INFO] YouTube URL detected — extracting stream via yt-dlp...")
        source_name = "youtube"
        source      = get_youtube_stream_url(source)
    # ─────────────────────────────────────────────────────────

    if source == "0" or (isinstance(source, str) and source.isdigit()):
        cap, source_name = cv2.VideoCapture(int(source)), "webcam"
    elif not is_youtube:
        cap, source_name = cv2.VideoCapture(source), Path(source).stem
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    writer = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{source_name}_detected.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out  = cap.get(cv2.CAP_PROP_FPS) or 30
        fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer   = cv2.VideoWriter(out_path, fourcc, fps_out, (fw,fh))
        print(f"  Saving to: {out_path}\n")

    frame_num = 0
    total     = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1

        results = model.predict(frame, conf=conf, iou=iou, imgsz=img_size, verbose=False)
        frame, count = draw_boxes(frame, results, class_names)
        total += count
        history.append(count)
        avg = sum(history) / len(history) if history else 0

        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        frame = draw_alert_flash(frame, avg, frame_num)
        frame = draw_hud(frame, fps, count, frame_num)

        cv2.imshow("PotholeGuard — CGIP  [Q to quit]", frame)
        if writer: writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit.")
            break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"\n  Frames: {frame_num}  |  Total detections: {total}\n")


# ─────────────────────────────────────────────
# Image mode
# ─────────────────────────────────────────────
def run_on_image(image_path, model_path, conf, iou, img_size, output_dir):
    model = YOLO(model_path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Not found: {image_path}")

    results = model.predict(frame, conf=conf, iou=iou, imgsz=img_size, verbose=False)
    frame, count = draw_boxes(frame, results, CONFIG["class_names"])

    avg = float(count)
    frame = draw_alert_flash(frame, avg, 0)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{Path(image_path).stem}_detected.jpg")
    cv2.imwrite(out, frame)
    print(f"[INFO] Saved: {out}  | Potholes: {count}")
    cv2.imshow("PotholeGuard — CGIP", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGIP PotholeGuard")
    parser.add_argument("--source",  type=str,   default="0")
    parser.add_argument("--model",   type=str,   default=CONFIG["model_path"])
    parser.add_argument("--conf",    type=float, default=CONFIG["conf_threshold"])
    parser.add_argument("--iou",     type=float, default=CONFIG["iou_threshold"])
    parser.add_argument("--imgsz",   type=int,   default=CONFIG["img_size"])
    parser.add_argument("--output",  type=str,   default=CONFIG["output_dir"])
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    src      = args.source
    img_exts = {".jpg",".jpeg",".png",".bmp",".webp"}

    if Path(src).suffix.lower() in img_exts:
        run_on_image(src, args.model, args.conf, args.iou, args.imgsz, args.output)
    else:
        run_detection(src, args.model, args.conf, args.iou,
                      args.imgsz, not args.no_save, args.output)
