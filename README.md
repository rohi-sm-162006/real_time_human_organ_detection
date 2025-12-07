## real_time_human_organ_detection

This repository hosts code and tooling to develop a real-time organ detection and segmentation system for laparoscopic videos (single-organ detection as a starting point).

Quick status
- Repository scaffold and stubs created. See `config/`, `data/`, `src/`, `notebooks/`, `scripts/`, `deployment/`, `tests/`.

Recommended public datasets
1. Cholec80 — laparoscopic cholecystectomy videos (good for surgical phase and tool presence; useful for sampling frames). See public dataset lists: https://github.com/luiscarlosgph/list-of-surgical-tool-datasets
2. Dresden Surgical Anatomy Dataset — pixel-wise organ annotations (8 organs + vessels). Great for training segmentation heads.
	- Download: https://gts.ai/dataset-download/the-dresden-surgical-anatomy-dataset/
3. Roboflow Laparoscopy (YOLO-ready) — labeled laparoscopic images preformatted for YOLO.
	- https://universe.roboflow.com/laparoscopic-yolo/laparoscopy
4. EndoVis challenge datasets (MICCAI) — surgical/endoscopic segmentation datasets with masks and benchmarks.
	- https://www.endovis.org
5. CT/MRI abdominal segmentation datasets for pretraining (transfer learning): Kaggle abdominal organ segmentation, AbdomenCT-1K.

High-level workflow
1. Data collection and annotation
	- Pull public datasets above. If using in-house/hospital videos, ensure IRB/ethics approval and de-identify videos.
	- Use `scripts/extract_frames.py` to sample frames from videos for annotation.
	- Annotate or convert annotations to per-frame mask PNGs (CVAT/Label Studio exports), placed in `data/annotations/` with corresponding images in `data/processed/`.
	- Convert masks to YOLOv8 segmentation layout with: `python scripts/convert_to_yolo.py` (this repository includes a placeholder — replace with polygon extraction for real labels).

2. Model training
	- Use `src/train_yolo.py` (Ultralytics YOLOv8) to train segmentation models. Configure `config/data.yaml` to point to your `data/yolov8` folder.
	- Recommended strategy:
	  - Pretrain backbone on CT/MRI organ datasets (optional).
	  - Train on laparoscopic datasets (Dresden, Roboflow images) and your annotated frames.
	  - Use augmentations (albumentations) to simulate smoke, blur, occlusions.

3. Inference and deployment
	- Use `src/demo_infer.py` to run a segmentation model on video files and save overlays.
	- For real-time on embedded devices (Jetson), convert the trained model to ONNX/TensorRT and use `deployment/jetson_infer.py` (stub provided).

Quick start
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
	(For GPU training, install the appropriate PyTorch+CUDA from https://pytorch.org)

2. Prepare data and labels, then convert:
	```bash
	python scripts/convert_to_yolo.py
	```

3. Train or use a pretrained segmenter:
	```bash
	python -m src.train_yolo
	```

4. Run demo inference:
	```bash
	python src/demo_infer.py /path/to/video.mp4 output.mp4
	```

Notes and next steps
- The current conversion script writes placeholder label files — implement a robust mask->polygon converter to generate valid YOLOv8 segmentation labels.
- Add CI to run unit tests and a small sample inference check.
- If you'd like, I can implement mask conversion, run a short demo with a public sample video, or prepare a GPU-ready Dockerfile.
