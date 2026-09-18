# Cityscapes Semantic Segmentation: CNN vs Transformers

### A Comparative Deep Learning Study of FCN, SegFormer, and TransFuse Architectures

This project explores **semantic image segmentation for autonomous driving** using six deep learning models from three architecture families: Fully Convolutional Networks (FCN), SegFormer, and TransFuse.

The models are trained and evaluated on the Cityscapes dataset to investigate how different architectural designs affect segmentation accuracy, computational efficiency, and inference speed.

The experiments include three baseline architectures and three modified variants, allowing direct comparison of CNN-based, Transformer-based, and hybrid approaches.

---

## Project Highlights

* Implemented and evaluated six semantic segmentation models.
* Developed a custom PyTorch dataset and preprocessing pipeline.
* Performed pixel-level classification across 19 Cityscapes classes.
* Compared CNN, Transformer, and hybrid architectures.
* Implemented architectural modifications to investigate performance differences.
* Evaluated models using Mean Intersection over Union (mIoU), pixel accuracy, parameter count, and inference time.

**Highest reported mIoU:** 49.49%, achieved by SegFormer MiT-B2.

**Highest reported pixel accuracy:** 92.18%, achieved by SegFormer MiT-B2.

---

## 1. Project Overview

Semantic segmentation assigns a class label to every pixel in an image.

Unlike image classification, which predicts a single label for an entire image, semantic segmentation provides a detailed understanding of the scene by identifying objects and regions at the pixel level.

This capability is particularly relevant to autonomous driving systems, where understanding roads, vehicles, pedestrians, traffic signs, and surrounding environments is essential.

The primary objective of this project is to compare three families of segmentation architectures and examine the effects of architectural modifications on performance.

### Models Implemented

| Architecture | Baseline         | Variant           |
| ------------ | ---------------- | ----------------- |
| FCN          | FCN-32s-style    | FCN-8s-style      |
| SegFormer    | MiT-B0           | MiT-B2            |
| TransFuse    | ViT-based fusion | Swin-based fusion |

The FCN implementations are custom ResNet50-based adaptations, while the TransFuse implementations are custom hybrid architectures inspired by CNN and Transformer feature fusion.

---

## 2. Dataset

**Dataset:** Cityscapes

**Source:** [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

**Kaggle dataset used:** [Cityscapes](https://www.kaggle.com/datasets/kavithak1388/cityscapes)

Cityscapes is a dataset designed for semantic understanding of urban street scenes.

It provides images of road environments with pixel-level annotations for objects and surrounding infrastructure.

### Dataset Statistics

| Property          | Value     |
| ----------------- | --------- |
| Training images   | 2,975     |
| Validation images | 500       |
| Number of classes | 19        |
| Input crop size   | 512 × 512 |
| Batch size        | 8         |
| Ignore index      | 255       |

### Segmentation Classes

The model predicts the following 19 classes:

```text
0  Road
1  Sidewalk
2  Building
3  Wall
4  Fence
5  Pole
6  Traffic Light
7  Traffic Sign
8  Vegetation
9  Terrain
10 Sky
11 Person
12 Rider
13 Car
14 Truck
15 Bus
16 Train
17 Motorcycle
18 Bicycle
```

Raw Cityscapes label IDs are mapped to consecutive training IDs from 0 to 18.

Pixels outside the selected classes are assigned an ignore index of 255.

---

## 3. Data Preprocessing

A custom `CityscapesDataset` class was implemented using PyTorch.

The preprocessing pipeline ensures that input images and segmentation masks undergo consistent spatial transformations.

### Training Transformations

**Random Scaling**

Images are randomly resized using a scale factor between 0.75 and 1.5.

**Random Horizontal Flip**

A horizontal flip is applied with a probability of 50%.

**Random Cropping**

Images and their corresponding segmentation masks are cropped to 512 × 512 pixels.

**Image Normalization**

Images are normalized using ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

**Label Mapping**

Original Cityscapes labels are converted to the 19 training classes.

Bilinear interpolation is used for image resizing, while nearest-neighbor interpolation is used for segmentation masks to preserve categorical labels.

---

# 4. Model Architectures

The project evaluates six models across three architecture families.

## 4.1 FCN Baseline

The baseline model implements a Fully Convolutional Network using a pretrained ResNet50 backbone.

The architecture consists of:

* ResNet50 feature extractor
* Convolutional classification head
* Bilinear upsampling
* Pixel-level class prediction

The encoder extracts high-level features, which are converted into segmentation predictions using a 1 × 1 convolution.

The resulting predictions are upsampled to the original input resolution.

**Reported results:**

* mIoU: 0.3870
* Pixel Accuracy: 87.92%

---

## 4.2 FCN Variant: FCN-8s

The FCN variant introduces skip connections to incorporate features from earlier encoder stages.

Unlike the baseline, which relies primarily on deep features, the variant combines information from multiple resolutions.

This allows the network to incorporate finer spatial details into its segmentation predictions.

### Architectural Modifications

* Added skip connections.
* Combined multi-scale feature representations.
* Improved spatial information recovery.

**Reported results:**

* mIoU: 0.4205
* Pixel Accuracy: 89.77%

The variant increased mIoU by 0.0335 compared with the FCN baseline.

---

## 4.3 SegFormer Baseline: MiT-B0

SegFormer is a Transformer-based semantic segmentation architecture that combines a hierarchical Transformer encoder with a lightweight segmentation head.

This implementation uses the pretrained MiT-B0 encoder.

The model extracts multi-scale features and combines them to produce segmentation predictions.

### Architecture

* MiT-B0 encoder
* Multi-scale feature extraction
* Lightweight segmentation decoder

**Reported results:**

* mIoU: 0.3962
* Pixel Accuracy: 89.72%

The model contains approximately 3.7 million trainable parameters.

---

## 4.4 SegFormer Variant: MiT-B2

The SegFormer variant replaces the MiT-B0 encoder with the larger MiT-B2 architecture.

This modification increases model capacity while retaining the overall SegFormer design.

### Architectural Modifications

* Replaced MiT-B0 with MiT-B2.
* Increased encoder capacity.
* Preserved the segmentation framework.

**Reported results:**

* mIoU: 0.4949
* Pixel Accuracy: 92.18%

The variant increased mIoU by 0.0987 compared with the MiT-B0 model.

It achieved the highest reported mIoU and pixel accuracy in this experiment.

---

## 4.5 TransFuse: CNN and Transformer Fusion

TransFuse combines convolutional feature extraction with Transformer-based representations.

The custom implementation uses two parallel branches:

**CNN Branch**

A pretrained ResNet50 extracts local spatial features.

**Transformer Branch**

A Vision Transformer (ViT) captures broader contextual information.

**Fusion Module**

A custom BiFusion module combines features from the two branches before generating segmentation predictions.

The objective is to combine the spatial characteristics of CNNs with the contextual representations produced by Transformers.

**Reported results:**

* mIoU: 0.4104
* Pixel Accuracy: 89.28%

---

## 4.6 TransFuse Variant: Swin Transformer

The TransFuse variant replaces the ViT branch with a Swin Transformer.

The CNN branch continues to use ResNet50.

The Swin Transformer introduces hierarchical representations through window-based self-attention.

The resulting features are combined with the CNN representations through the fusion architecture.

### Architectural Modifications

* Replaced ViT with Swin Transformer.
* Retained the ResNet50 CNN branch.
* Preserved the hybrid feature fusion design.

**Reported results:**

* mIoU: 0.4050
* Pixel Accuracy: 89.04%

The variant reduced parameter count and inference time, but its reported mIoU was slightly lower than that of the original TransFuse implementation.

---

# 5. Training Configuration

All six models were trained using PyTorch.

The training pipeline includes forward propagation, loss calculation, backpropagation, optimizer updates, and validation.

| Parameter               | Configuration      |
| ----------------------- | ------------------ |
| Epochs                  | 20                 |
| Batch Size              | 8                  |
| Input Size              | 512 × 512          |
| Optimizer               | Adam               |
| Loss Function           | Cross-Entropy Loss |
| Ignore Index            | 255                |
| Learning Rate Scheduler | PolynomialLR       |
| Random Seed             | 42                 |

### Learning Rates

| Model            | Learning Rate |
| ---------------- | ------------- |
| FCN              | 1e-4          |
| FCN-8s           | 1e-4          |
| SegFormer MiT-B0 | 6e-5          |
| SegFormer MiT-B2 | 6e-5          |
| TransFuse        | 1e-4          |
| TransFuse Swin   | 1e-4          |

The training pipeline saves model checkpoints based on validation mIoU.

---

# 6. Evaluation Metrics

Four metrics are used to compare the models.

### Mean Intersection over Union (mIoU)

Mean IoU measures the overlap between predicted segmentation regions and ground-truth regions.

For an individual class:

$$
IoU = \frac{TP}{TP + FP + FN}
$$

The metric is averaged across the evaluated classes.

### Pixel Accuracy

Measures the proportion of correctly classified pixels among valid pixels.

$$
Pixel\ Accuracy =
\frac{\text{Correct Pixels}}{\text{Valid Pixels}}
$$

### Parameter Count

Measures the number of trainable parameters in each model.

This provides an indication of model size and computational complexity.

### Inference Time

Measures the time required for the model to perform inference on an input image.

The notebook benchmarks inference using a 512 × 512 input.

**Evaluation note:** The notebook averages batch-level mIoU and pixel accuracy rather than computing a single dataset-wide confusion matrix. This distinction should be considered when comparing these results with externally published Cityscapes benchmarks.

---

# 7. Experimental Results

The following results were reported in the project notebook.

| Model            |       mIoU | Pixel Accuracy | Parameters | Inference Time |
| ---------------- | ---------: | -------------: | ---------: | -------------: |
| FCN Baseline     |     0.3870 |         87.92% |     23.55M |        18.8 ms |
| FCN-8s           |     0.4205 |         89.77% |     23.58M |        17.6 ms |
| SegFormer MiT-B0 |     0.3962 |         89.72% |      3.72M |        16.3 ms |
| SegFormer MiT-B2 | **0.4949** |     **92.18%** |     27.36M |        59.6 ms |
| TransFuse ViT    |     0.4104 |         89.28% |    111.08M |        32.9 ms |
| TransFuse Swin   |     0.4050 |         89.04% |     52.21M |        23.5 ms |

## Key Findings

**SegFormer MiT-B2**

Achieved the highest reported mIoU and pixel accuracy, although it also had the longest measured inference time.

**FCN-8s**

The addition of skip connections increased mIoU from 0.3870 to 0.4205.

**SegFormer MiT-B0**

Had the fewest parameters and the shortest reported inference time.

**TransFuse Swin**

Reduced the parameter count from approximately 111 million to 52 million and lowered inference time from 32.9 ms to 23.5 ms, with a small reduction in mIoU.

These findings illustrate the trade-offs between segmentation accuracy, architectural complexity, and inference speed.

---

# 8. Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* Torchvision
* Hugging Face Transformers

### Data Processing

* NumPy
* PIL / Pillow
* Python Glob

### Visualization and Analysis

* Matplotlib
* Pandas

### Development Environment

* Kaggle Notebooks
* CUDA GPU

---

# 9. Project Structure

```text
cityscapes-semantic-segmentation/
│
├── README.md
│
└── DL - Yara 236125.ipynb
    │
    ├── Dataset Preprocessing
    ├── Evaluation Metrics
    ├── FCN Baseline
    ├── SegFormer MiT-B0
    ├── TransFuse ViT
    ├── FCN-8s Variant
    ├── SegFormer MiT-B2 Variant
    ├── TransFuse Swin Variant
    └── Results Comparison
```

The repository contains the original experiment notebook with model implementations, training loops, evaluation procedures, and results.

---

# 10. Getting Started

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cityscapes-semantic-segmentation.git

cd cityscapes-semantic-segmentation
```

Replace `YOUR_USERNAME` with your GitHub username.

## Step 2: Install Dependencies

```bash
pip install torch torchvision transformers numpy pandas pillow matplotlib
```

A CUDA-compatible GPU is recommended for training.

Library versions are not pinned in the original notebook, so adjustments may be required when using newer releases.

## Step 3: Download the Dataset

Download the Cityscapes dataset using the Kaggle link provided above.

The notebook expects the dataset to follow this structure:

```text
Cityscape/
│
├── leftImg8bit/
│   ├── train/
│   └── val/
│
└── gtFine/
    ├── train/
    └── val/
```

Update the dataset root path in the notebook:

```python
ROOT = "/path/to/Cityscape"
```

## Step 4: Run the Notebook

Open the notebook in Kaggle or Jupyter Notebook.

Execute the cells sequentially to:

1. Load and preprocess the dataset.
2. Initialize the segmentation models.
3. Train each model.
4. Evaluate validation performance.
5. Save checkpoints.
6. Compare the experimental results.

---

# 1w. Author

**Yara Hatem**

Deep Learning Project

Focus: Semantic Image Segmentation and Comparative Deep Learning Architectures

---

# 14. References

* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
* [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
* [TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation](https://arxiv.org/abs/2102.08005)
* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

---

## License

No license has been specified for this repository. Any future license should account for the Cityscapes dataset terms and the licenses of pretrained model weights used in the project.
