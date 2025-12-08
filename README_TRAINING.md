# Training with YOLOv8 (segmentation)
1. Prepare data: run `python scripts/convert_to_yolo.py` to build `data/yolov8/`.
2. Install dependencies: `pip install -r requirements.txt` (ultralytics included).

3. Train: `python -m src.train_yolo --data config/data.yaml --epochs 50` or use the script directly.
