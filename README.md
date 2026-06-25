# Pedestrian Detection and Anomaly Detection Pipeline
---

## Overview

```
Train on MOT17 (normal pedestrian patterns)
         │
    YOLOv5 Detection
    (YOLOv5s / YOLOv5m / YOLOv5m-Frozen compared)
         │
  BoT-SORT / DeepSORT Multi-Object Tracking
  (Kalman Filter + Re-ID appearance features)
         │
  Rule-Based Anomaly Logic
  (Speed │ Direction Change │ Forbidden Zones)
         │
  Anomaly Flagging & Annotated Video Export
```

---

---

## Phase I — Object Detection on MOT17

### Dataset

The **Multiple Object Tracking 2017 (MOT17)** dataset provides 14 video sequences (7 train, 7 test) filmed in crowded public spaces. Annotations follow the format `frame, id, x, y, w, h, conf, cls, vis`. Only pedestrian class (cls=1) annotations with visibility ≥ 0.3 are used. Every 3rd frame is sampled to keep the dataset manageable; a 15% split is reserved for validation.

| Sequence | Environment | Challenge |
|---|---|---|
| MOT17-02 | Shopping Mall | Large crowds, static camera |
| MOT17-04 | Town Square | Night-time, low light |
| MOT17-09 | Pedestrian Street | Low angle, high occlusion |
| MOT17-10 | Public Square | Distant pedestrians (small objects) |

### Data Preprocessing

MOT17 bounding boxes (`top-left x, y, width, height`) are converted to YOLO's normalised centre format:

```python
cx = (x + w / 2) / img_w
cy = (y + h / 2) / img_h
w_norm = w / img_w
h_norm = h / img_h
```

### Model Variants

Three YOLOv5 configurations are evaluated:

| Model | Parameters | Training Strategy |
|---|---|---|
| YOLOv5s | ~7.2 M | All layers trained from COCO pre-train |
| YOLOv5m | ~21.2 M | All layers trained from COCO pre-train |
| YOLOv5m-Frozen | ~21.2 M | Backbone (layers 0–9) frozen; only neck + head fine-tuned |

**Transfer Learning rationale:** Freezing the backbone retains low-level COCO features (edges, textures, shapes) while adapting only the detection head to the MOT17 pedestrian class. This prevents catastrophic forgetting and accelerates convergence when the target dataset is smaller than the original pre-training corpus.

### Results — 30 Epochs

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv5s | 0.953 | 0.873 | 0.956 | 0.696 |
| YOLOv5m | **0.959** | **0.911** | **0.968** | **0.753** |
| YOLOv5m-Frozen | 0.947 | 0.885 | 0.961 | 0.717 |


---

## Phase II — Multi-Object Tracking & Anomaly Detection

### Tracking Algorithm

The **BoT-SORT** tracker (an enhancement of DeepSORT) is integrated with the YOLOv5m-Frozen detector. Key components:

- **Kalman Filter** — predicts each track's position and velocity between frames, maintaining identity through occlusion.
- **Hungarian Algorithm** — solves the assignment problem between predicted track locations and new detections using IoU cost: `cost = 1 − IoU(box_predicted, box_detected)`.
- **Re-ID Network** — appearance embeddings from a MobileNet backbone reduce identity switches when two pedestrians cross paths.
- **Camera-motion compensation** — BoT-SORT's improvement over vanilla DeepSORT; keeps tracks stable on moving-camera sequences.

**Class filtering:** During initial testing the tracker produced false positives on background objects (monitors, furniture) due to the COCO-pre-trained backbone. Passing `classes=[0]` restricts inference to the Person class only, eliminating environmental noise without any retraining.

```python
results = model.track(
    source="mot_sample.mp4",
    conf=0.5,
    persist=True,
    classes=[0]
)
```

### Inference Dataset — CUHK Avenue

The **CUHK Avenue Dataset** consists of fixed overhead surveillance footage of a building avenue containing both normal and abnormal pedestrian behaviour (running, throwing objects, walking in wrong directions). It is used exclusively for inference — the detector is not retrained on Avenue.

### Anomaly Detection Rules

Anomalies are detected via three rule-based checks applied per track at each frame:

| Rule | Threshold | Flag |
|---|---|---|
| Speed | > 40 px/frame | `SPEED(value)` |
| Direction change | > 120° heading reversal | `DIR(degrees)` |
| Forbidden zone | Centroid enters defined region | `FORBIDDEN_ZONE` |

This unsupervised, rule-based approach works because the detector is trained only on normal MOT17 pedestrian patterns. Deviations in motion profile — speed, heading, or spatial boundary — are treated as anomalies, which mirrors realistic deployment where labelled anomaly data is scarce.


---

## Key Findings

1. **YOLOv5m is the top performer overall** (mAP@0.5 = 0.968 at 30 epochs), but requires ~21 M parameters and full fine-tuning time.
2. **YOLOv5m-Frozen converges faster** (≈10 minutes for 5 epochs) and reaches strong accuracy by preventing catastrophic forgetting of COCO features. It is the recommended choice when compute time or dataset size is limited.
3. **YOLOv5s** remains a viable baseline for edge deployments where inference speed matters more than the last few mAP points.
4. **Class filtering** (`classes=[0]`) is a simple but critical post-processing step that eliminates environmental false positives without any model changes.
5. **Rule-based anomaly detection** on tracked trajectories is effective without requiring labelled anomaly training data — the system relies entirely on deviations from normal motion statistics.

