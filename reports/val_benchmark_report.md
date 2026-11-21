# Step 4 Validation Benchmark Report (YOLOv8n vs. YOLO11n)

> **Generated (UTC):** 2026-11-21 15:30:40 UTC
> **Evaluation Policy:** Evaluated strictly on the **validation split (`val`)**. The held-out `test` split was **NOT** touched in this step.

## 1. Validation Split Composition
- **Total images:** 547
- **Single-item (`1 box`):** 427
- **Multi-class (`>=2 distinct classes`):** 69
- **Same-class multi-instance (`>=2 boxes, 1 class`):** 51

## 2. Model Comparison (Validation Set)

| Model | Precision | Recall | mAP@50 | mAP@50-95 | Best Conf | Balanced F1 | FP / img |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n** | 0.6880 | 0.5458 | 0.6276 | **0.4465** | 0.30 | 0.5641 | 0.4936 |
| **yolo11n** | 0.6861 | 0.5887 | 0.6532 | **0.4753** | 0.25 | 0.5728 | 0.5393 |

## 3. Single-Image (`batch=1`) Inference Latency (Target `< 100 ms`)

| Model | Primary Device | Mean (ms) | Median (ms) | P95 (ms) | CPU Mean (ms) | Meets Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8n** | 0 | 10.66 | 8.03 | 17.50 | 26.06 | PASS |
| **yolo11n** | 0 | 11.34 | 8.58 | 17.18 | 24.66 | PASS |

## 4. Selected Winner
- **Winning Model:** `yolo11n` (`mAP@50-95 = 0.4753`, `mAP@50 = 0.6532`)
- **Recommended Confidence Threshold:** `0.25`
- **Exported Checkpoint:** `models/best.pt`
- **Metadata:** `models/model_meta.json`

## 5. Confidence Threshold Sweep — `yolo11n` (`val` set, IoU >= 0.50)

| Conf | Precision | Recall | F1 | Multi-Item Recall | Multi-Item F1 | FP / img | FN / img |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.15 | 0.5068 | 0.6489 | 0.5691 | 0.5342 | 0.5333 | 0.8647 | 0.4808 |
| 0.20 | 0.5567 | 0.6288 | 0.5906 | 0.5000 | 0.5421 | 0.6856 | 0.5082 |
| 0.25 | 0.6072 | 0.6088 | 0.6080 | 0.4658 | 0.5376 | 0.5393 | 0.5356 |
| 0.30 | 0.6391 | 0.5888 | 0.6129 | 0.4410 | 0.5308 | 0.4552 | 0.5631 |
| 0.35 | 0.6656 | 0.5661 | 0.6118 | 0.4130 | 0.5175 | 0.3894 | 0.5941 |
| 0.40 | 0.7088 | 0.5461 | 0.6169 | 0.3882 | 0.5051 | 0.3071 | 0.6216 |
| 0.45 | 0.7486 | 0.5247 | 0.6170 | 0.3665 | 0.4927 | 0.2413 | 0.6508 |
| 0.50 | 0.7844 | 0.4953 | 0.6072 | 0.3354 | 0.4685 | 0.1865 | 0.6910 |
| 0.55 | 0.8114 | 0.4766 | 0.6005 | 0.3168 | 0.4513 | 0.1517 | 0.7166 |
| 0.60 | 0.8166 | 0.4459 | 0.5769 | 0.2919 | 0.4253 | 0.1371 | 0.7587 |
