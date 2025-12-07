"""Demo inference: run a YOLOv8 segmentation model on a video and save overlays."""
import cv2
import os
import numpy as np

def overlay_segmentation(frame, masks, boxes=None, labels=None, colors=None):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        color = (0, 255, 0) if not colors else tuple(colors[i % len(colors)])
        colored = np.zeros_like(frame, dtype=np.uint8)
        colored[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)
    if boxes:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if labels:
                cv2.putText(overlay, labels[i], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return overlay

def run_demo(video_path, out_path='output.mp4', model_path='yolov8n-seg.pt', device='cpu'):
    try:
        from ultralytics import YOLO
    except Exception:
        print('Please install ultralytics: pip install ultralytics')
        return

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Cannot open video', video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, device=device)[0]
        masks = []
        boxes = []
        labels = []
        if hasattr(results, 'masks') and results.masks is not None:
            # results.masks.xy or results.masks.data
            try:
                mask_arr = results.masks.data if hasattr(results.masks, 'data') else results.masks
                for m in mask_arr:
                    masks.append((m.cpu().numpy() > 0.5).astype('uint8'))
            except Exception:
                masks = []
        if hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes.data:
                x1, y1, x2, y2 = box[:4]
                boxes.append([x1, y1, x2, y2])
        if hasattr(results, 'names'):
            labels = [results.names.get(int(c), 'org') for c in getattr(results, 'cls', [])]

        out_frame = overlay_segmentation(frame, masks, boxes=boxes, labels=labels)
        out.write(out_frame)

    cap.release()
    out.release()
    print('Saved annotated video to', out_path)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/demo_infer.py /path/to/video.mp4 [out.mp4]')
    else:
        run_demo(sys.argv[1], out_path=(sys.argv[2] if len(sys.argv) > 2 else 'output.mp4'))
