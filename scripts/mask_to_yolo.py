"""Convert per-pixel mask PNGs into YOLOv8 polygon label files.

Assumptions:
- Each image in `images_dir` has a corresponding mask PNG in `masks_dir` with the same base name.
- Masks are single-channel where different integer values represent class ids (0 = background).
- Output is written to `out_dir/images` and `out_dir/labels` in YOLOv8 polygon label format:
  class_id x1 y1 x2 y2 ... xn yn  (normalized coordinates 0..1)
"""
import os
import cv2
import numpy as np


def mask_to_polygons(mask: np.ndarray, class_id=0, epsilon_frac=0.01, min_area=100):
    # mask: 2D array where mask>0 indicates object
    contours, _ = cv2.findContours((mask).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    h, w = mask.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        epsilon = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        coords = approx.reshape(-1, 2)
        # Normalize
        norm = coords.astype(float)
        norm[:, 0] /= w
        norm[:, 1] /= h
        # Flatten to x1 y1 x2 y2 ...
        if norm.size == 0:
            continue
        polys.append((class_id, norm.flatten()))
    return polys


def convert_dir(images_dir='data/processed', masks_dir='data/annotations', out_dir='data/yolov8'):
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
            print('mask missing for', fname)
            continue

        img = cv2.imread(img_path)
        if img is None:
            print('cannot read image', img_path)
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print('cannot read mask', mask_path)
            continue

        # If multi-channel, convert to single channel
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        # If mask contains multiple class ids, iterate unique ids > 0
        unique_vals = np.unique(mask_gray)
        labels = []
        for val in unique_vals:
            if val == 0:
                continue
            single_mask = (mask_gray == val).astype('uint8') * 255
            polys = mask_to_polygons(single_mask, class_id=int(val))
            labels.extend(polys)

        # Save image
        out_img = os.path.join(images_out, fname)
        cv2.imwrite(out_img, img)

        # Write label file with YOLOv8 polygon format
        label_file = os.path.join(labels_out, base + '.txt')
        with open(label_file, 'w') as f:
            for class_id, poly in labels:
                # YOLOv8 polygon expects: class x1 y1 x2 y2 ...
                parts = [str(int(class_id))] + [f"{x:.6f}" for x in poly.tolist()]
                f.write(' '.join(parts) + '\n')


if __name__ == '__main__':
    convert_dir()
