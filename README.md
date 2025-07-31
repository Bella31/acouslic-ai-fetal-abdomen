# Acoustic AI Fetal Abdomen Analysis

A comprehensive deep learning pipeline for analyzing fetal abdomen ultrasound scans using acoustic AI techniques. This project consists of two main components: frame classification and abdomen segmentation.

## Project Overview

This project implements a two-stage pipeline for fetal abdomen ultrasound analysis:

1. **Part 1: Frame Classification** - Identifies optimal, suboptimal, and irrelevant frames from ultrasound scans
2. **Part 2: Abdomen Segmentation** - Segments the fetal abdomen region from optimal/suboptimal frames

## Project Structure

```
acouslic-ai-fetal-abdomen/
├── part1_frame_classification/
│   ├── requirements_part1.txt
│   ├── scripts/
│   │   ├── calculate_wfss.py
│   │   ├── evaluate_frame_selection.py
│   │   ├── extract_frames_from_mha.py
│   │   └── prepare_balanced_dataset.py
│   └── src/
│       ├── model_resnet.py
│       ├── train_classification.py
│       └── utils.py
├── part2_abdomen_segmentation/
│   ├── requirements_part2.txt
│   ├── scripts/
│   │   ├── evaluate_segmentation.py
│   │   ├── prepare_segmentation_dataset.py
│   │   └── split_segmentation_dataset.py
│   └── src/
│       ├── model_unet.py
│       ├── train_segmentation.py
│       └── utils.py
└── README.md
```

## Part 1: Frame Classification

### Overview
The frame classification component uses a fine-tuned ResNet-50 model to classify ultrasound frames into three categories:
- **Optimal**: High-quality frames suitable for analysis
- **Suboptimal**: Acceptable frames with some quality issues
- **Irrelevant**: Poor quality frames that should be excluded

### Key Features
- **ResNet-50 Architecture**: Pre-trained model fine-tuned for grayscale ultrasound images
- **Data Augmentation**: Comprehensive augmentation including rotation, scaling, and color jittering
- **Mixup Training**: Advanced training technique to improve generalization
- **Focal Loss**: Handles class imbalance in medical imaging data
- **Early Stopping**: Prevents overfitting with patience-based stopping

### Installation
```bash
cd part1_frame_classification
pip install -r requirements_part1.txt
```

### Usage

#### 1. Extract Frames from MHA Files
```bash
python scripts/extract_frames_from_mha.py \
    --csv path/to/labels.csv \
    --mha_dir path/to/mha/scans \
    --output path/to/output/dataset \
    --train_ratio 0.7 \
    --val_ratio 0.15
```

#### 2. Train Classification Model
```bash
python src/train_classification.py \
    --data_path path/to/dataset \
    --epochs 50 \
    --batch_size 32 \
    --model_save_path models/best_classifier.pt
```

#### 3. Evaluate Frame Selection
```bash
python scripts/evaluate_frame_selection.py \
    --scan_dir path/to/mha/scans \
    --output results/predictions.csv \
    --model models/best_classifier.pt \
    --val_scans path/to/val_scans.csv
```

#### 4. Calculate WFSS Score
```bash
python scripts/calculate_wfss.py \
    --predictions results/predictions.csv \
    --mask_dir path/to/masks
```

## Part 2: Abdomen Segmentation

### Overview
The segmentation component uses a U-Net architecture to segment the fetal abdomen region from ultrasound frames. This is trained on frames classified as optimal or suboptimal from Part 1.

### Key Features
- **U-Net Architecture**: Classic encoder-decoder structure with skip connections
- **Binary Segmentation**: Focuses on abdomen vs. background classification
- **Multiple Metrics**: Dice coefficient, Hausdorff distance, and normalized AC error
- **Data Preprocessing**: Comprehensive transforms for medical imaging

### Installation
```bash
cd part2_abdomen_segmentation
pip install -r requirements_part2.txt
```

### Usage

#### 1. Prepare Segmentation Dataset
```bash
python scripts/prepare_segmentation_dataset.py \
    --frames_folder1 path/to/optimal/frames \
    --frames_folder2 path/to/suboptimal/frames \
    --mask_dir path/to/mha/masks \
    --output_img_dir path/to/segmentation/images \
    --output_mask_dir path/to/segmentation/masks
```

#### 2. Split Dataset
```bash
python scripts/split_segmentation_dataset.py \
    --images_dir path/to/segmentation/images \
    --masks_dir path/to/segmentation/masks \
    --output_dir path/to/split/dataset \
    --train_ratio 0.7 \
    --val_ratio 0.15
```

#### 3. Train Segmentation Model
```bash
python src/train_segmentation.py \
    --train_images path/to/train/images \
    --train_masks path/to/train/masks \
    --val_images path/to/val/images \
    --val_masks path/to/val/masks \
    --epochs 50 \
    --batch_size 8 \
    --save_path models/unet_best.pth
```

#### 4. Evaluate Segmentation
```bash
python scripts/evaluate_segmentation.py \
    --model_path models/unet_best.pth \
    --test_images path/to/test/images \
    --test_masks path/to/test/masks \
    --scan_dir path/to/mha/scans
```

## Model Architectures

### Frame Classification (ResNet-50)
- **Input**: 256x256 grayscale ultrasound frames
- **Output**: 3-class classification (optimal/suboptimal/irrelevant)
- **Modifications**: First convolutional layer adapted for single-channel input
- **Training**: Fine-tuning with mixup and focal loss

### Abdomen Segmentation (U-Net)
- **Input**: 256x256 grayscale ultrasound frames
- **Output**: Binary segmentation mask (abdomen/background)
- **Architecture**: 4-level encoder-decoder with skip connections
- **Loss**: Combined BCE and Dice loss

## Evaluation Metrics

### Frame Classification
- **Accuracy**: Standard classification accuracy
- **WFSS (Weighted Frame Selection Score)**: Custom metric for frame selection quality
  - Optimal frame selected: 1.0
  - Suboptimal frame selected (when optimal available): 0.6
  - Irrelevant frame selected: 0.0

### Abdomen Segmentation
- **Dice Coefficient**: Measures overlap between predicted and ground truth masks
- **Hausdorff Distance**: Measures boundary accuracy
- **Normalized AC Error**: Measures area consistency error
- **Combined Score**: Average of normalized metrics

## Data Requirements

### Input Format
- **MHA Files**: 3D ultrasound volumes (.mha format)
- **CSV Labels**: Frame-level annotations with columns: Filename, Frame, Label
- **Masks**: Binary segmentation masks in MHA format

### Data Organization
```
dataset/
├── train/
│   ├── optimal/
│   ├── suboptimal/
│   └── irrelevant/
├── val/
│   ├── optimal/
│   ├── suboptimal/
│   └── irrelevant/
└── test/
    ├── optimal/
    ├── suboptimal/
    └── irrelevant/
```

## Dependencies

### Part 1 Requirements
- torch>=2.0.0
- torchvision>=0.15.0
- pandas
- numpy
- Pillow
- scikit-learn
- tqdm
- SimpleITK

### Part 2 Requirements
- torch
- torchvision
- pandas
- numpy
- opencv-python
- SimpleITK
- scikit-learn
- scipy
- tqdm
- Pillow

## Performance Considerations

- **GPU Recommended**: Both models benefit significantly from GPU acceleration
- **Memory Requirements**: U-Net training requires more memory than classification
- **Batch Size**: Adjust based on available GPU memory
- **Mixed Precision**: Automatic mixed precision training for efficiency

## Contributing

This project is designed for medical imaging research. When contributing:
1. Ensure all code follows medical imaging best practices
2. Validate results on appropriate test datasets
3. Document any changes to model architectures or evaluation metrics
4. Test thoroughly before deployment in clinical settings
