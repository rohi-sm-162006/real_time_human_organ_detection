"""Convert polygon/mask annotations into YOLOv8 segmentation format.

This script assumes annotations are provided as per-image mask PNGs in
`data/annotations/` and images in `data/processed/`.
Outputs a `data/yolov8/` directory with images and YOLO segmentation labels.
"""
import os
from PIL import Image

def convert_masks_to_yolo(images_dir, masks_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    images_out = os.path.join(out_dir, 'images')
    labels_out = os.path.join(out_dir, 'labels')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, base + '.png')
        if not os.path.exists(mask_path):
            continue
        # copy image
        Image.open(img_path).save(os.path.join(images_out, fname))
        # write simple YOLO segmentation label placeholder
        label_file = os.path.join(labels_out, base + '.txt')
        # For pixel masks you'd convert polygons to RLE or normalized xy coords;
        # here we produce a placeholder line per object: class_id and a dummy polygon.
        with open(label_file, 'w') as f:
            f.write('0 0.0 0.0 1.0 1.0\n')

if __name__ == '__main__':
    convert_masks_to_yolo('data/processed', 'data/annotations', 'data/yolov8')
