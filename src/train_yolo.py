"""YOLOv8 segmentation training runner (uses ultralytics API)"""
import os

def train_yolov8(data_yaml='config/data.yaml', model='yolov8n-seg.pt', epochs=50):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('Ultralytics package not installed. Please pip install ultralytics')
        raise

    yolo = YOLO(model)
    yolo.train(data=data_yaml, epochs=epochs)

if __name__ == '__main__':
    train_yolov8()
