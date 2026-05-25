from ultralytics import YOLO
import cv2

# ✅ FIX 1: Use forward slash (safer)
model = YOLO("C:/Users/sksaa/OneDrive/Desktop/smartVision AI/notebook/runs/detect/yolo_custom-7/weights/best.pt")

def predict_image(image_path, conf=0.5):

    results = model(image_path, conf=conf)

    result = results[0]
    boxes = result.boxes
    names = model.names

    detections = []

    # ✅ FIX 2: Handle no detections case
    if boxes is None:
        return [], result.plot()

    for box in boxes:
        cls = int(box.cls[0])
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf_score >= conf:   # >= better
            detections.append({
                "label": names[cls],
                "confidence": round(conf_score, 2),
                "box": [x1, y1, x2, y2]
            })

    output_img = result.plot()

    return detections, output_img