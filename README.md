# Intelligent Road Damage Detection — Anveshan DataSprint 2.0

Comprehensive deep learning solution for automated detection and classification of road damage: Longitudinal Cracks (D00), Transverse Cracks (D10), Alligator Cracks (D20), and Potholes (D40).

## 1. Executive Summary
- **Competition Track**: Object Detection (Road Damage)
- **Primary Metric**: Weighted Leaderboard Score (mAP@50, mAP@50:95, Precision, Recall, Speed, Robustness)
- **Public Leaderboard Score**: **39.38940**
- **Reproduced Validation Score**: **40.23 / 100** (Accuracy score within 0.84 points of public score)
- **Inference Speed**: ~5.2 ms / image on NVIDIA Tesla T4 GPU (~3.5 minutes total runtime across 4,200 test photos)

## 2. Model Architecture
- **Base Architecture**: YOLOv8 Nano (`yolov8n.pt`)
- **Pretrained Weights**: COCO (generic weights, strictly approved)
- **Input Resolution**: 640 × 640
- **Number of Classes**: 4 (`0: D00`, `1: D10`, `2: D20`, `3: D40`)

## 3. Exploratory Data Analysis & Preprocessing
- **Dataset Size**: 14,691 training images, 2,150 validation images, 4,200 test images.
- **Label Integrity**: 17,517 training bounding boxes. Zero inverted coordinates, zero NaN values.
- **Background Samples**: Kept all clean road images (~42% background images without damage) to train the model to suppress false positives on undamaged asphalt.
- **Coordinate Transformation**: Converted absolute pixel coordinates $(x_1, y_1, x_2, y_2)$ to normalized YOLO format $(x_c, y_c, w, h)$.

## 4. Validation Strategy
- Leakage-free validation using the official 2,150 image split (`val_labels.csv`).
- Strict image isolation verified: 0 image overlap between train and validation sets.

## 5. Training Setup & Hyperparameters
- **Optimizer**: SGD with momentum 0.937 and weight decay 0.0005
- **Epochs**: 25 epochs
- **Batch Size**: 16
- **Augmentation**: Horizontal flip ($p=0.5$), HSV adjustment, scale augmentation ($0.5$).
- **Early Stopping**: Disabled / high patience to ensure full convergence.

## 6. Inference Pipeline & Contract (`inference.py`)
- Standardized entry points: `load(device)` and `predict(model, image)`.
- **Confidence Cutoff**: Preserved detections down to `conf=0.001` to maximize mAP area under the precision-recall curve.
- **Non-Maximum Suppression (NMS)**: IoU threshold 0.60, `max_det=100`.
- **Edge Guard**: Automatic clipping to image boundaries $[0, W] \times [0, H]$ and degenerate box rejection ($w \ge 1, h \ge 1$).

## 7. Speed Benchmark & Resource Profiling
- Execution time per image: **5.2 ms** on Tesla T4 GPU.
- Total processing time for 4,200 test images: **~3.5 minutes** (well under the 30-minute threshold).
- Speed score: **1.00 / 1.00** ($t \le 25\text{ms}$).

## 8. Validation Evaluation Results
Evaluation on 2,150 validation images using `score.py`:
- **mAP@50**: `0.4770`
- **mAP@50:95**: `0.2008`
- **Precision**: `0.5179` (at optimal F1 confidence cutoff 0.182)
- **Recall**: `0.4732`
- **Per-Class Breakdown**:
  - D00 (Longitudinal): `AP50 = 0.4216`
  - D10 (Transverse): `AP50 = 0.3968`
  - D20 (Alligator): `AP50 = 0.6369`
  - D40 (Pothole): `AP50 = 0.4528`
- **Accuracy Score**: **40.23 / 100**

## 9. Reproducibility Instructions
To reproduce the exact submission and score in under 15 minutes:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train_baseline.py --kit . --model yolov8n.pt --epochs 25 --imgsz 640 --batch 16

# 3. Benchmark and generate predictions
python benchmark_speed.py --repo . --images test_images --out-csv test_pred.csv --out-json speed.json

# 4. Create submission CSV
python make_kaggle_submission.py --pred test_pred.csv --image-ids test_image_ids.csv --out submission.csv

# 5. Verify compliance
python check_submission.py --pred submission.csv --image-ids test_image_ids.csv
```

## 10. Submission Format & Diagnostics
- Format: `image_id,PredictionString`
- Images with zero detections are labeled `none` as required.
- Total test images processed: 4,200 rows with 0 missing IDs.

## 11. What Worked vs. What Failed
- **What Worked**:
  - Training on full original resolutions with normalized coordinates.
  - Retaining low-confidence detections ($0.001$) to maximize mAP interpolation.
  - Keeping background images to optimize precision on undamaged pavement.
- **What Failed**:
  - Aggressive early stopping at epoch 4 (caused severe underfitting and 15.78 score).
  - Heavy mixup and copy-paste augmentations from scratch without pre-warming (confused crack features).

## 12. Licensing & Ethics
- Dataset: RDD2020 licensed under CC BY-NC 3.0.
- Pretrained weights: Ultralytics YOLOv8n pretrained exclusively on MS COCO.
- No external road damage data or manual test annotations were used.
