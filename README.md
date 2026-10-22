# 🧊 Attention U-Net Multispectral Glacier Segmentation Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform: Windows | Linux | Colab](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Colab-green.svg)]()

An independent, production-grade Deep Learning framework implementing an **8-Channel Attention U-Net** for automated, high-precision glacier and glacial lake segmentation in rugged alpine terrain using Copernicus Sentinel-2 multispectral imagery and Randolph Glacier Inventory (RGI) reference labels.

- **Academic Context**: B.Tech Information Technology (7th Semester Minor Project), National Institute of Technology (NIT) Raipur.
- **Author**: Piyush Verma, Deepanshu Sahu, Rameshwar Chandravanshi.

---

## 📑 Table of Contents

1. [Project Overview & Motivation](#-project-overview--motivation)
2. [Problem Statement & Scientific Challenges](#-problem-statement--scientific-challenges)
3. [The 8-Channel Multispectral Input Pipeline](#-the-8-channel-multispectral-input-pipeline)
4. [Attention U-Net Architecture & Attention Gates](#-attention-u-net-architecture--attention-gates)
5. [Loss Function & Optimization Mechanics](#-loss-function--optimization-mechanics)
6. [Evaluation Metrics & Threshold Tuning](#-evaluation-metrics--threshold-tuning)
7. [Repository Structure](#-repository-structure)
8. [Installation & Local Setup](#-installation--local-setup)
9. [Running Tests, Training, Evaluation & Inference](#-running-tests-training-evaluation--inference)
10. [Google Colab Workflow](#-google-colab-workflow)
11. [Data Attribution & References](#-data-attribution--references)
12. [License](#-license)

---

## 🔭 1. Project Overview & Motivation

Manual delineation of glaciers and glacial lakes from satellite imagery is time-consuming, subjective, and practically infeasible at regional scales due to the sheer volume of high-resolution remote-sensing data. Climate change accelerates glacial retreat and expands dangerous glacial lake outburst flood (GLOF) hazards in high mountain ranges such as the Himalayas.

This project builds a fully automated computer vision and remote-sensing pipeline that ingests orbital satellite observations, computes physical cryospheric indices, and uses a state-of-the-art **Attention U-Net** to segment ice bodies with pixel-level precision.

---

## 🏔️ 2. Problem Statement & Scientific Challenges

Standard deep learning segmentation models trained on natural 3-channel RGB imagery struggle in high-altitude terrain due to:

- **Debris-Covered Ice**: Supraglacial debris mimics the spectral signature of surrounding moraines and barren rock walls.
- **Topographic Shadows**: Deep mountain shadows obscure clean ice boundaries in standard optical bands.
- **Seasonal Snow Noise**: Transient seasonal snow packs confuse basic thresholding algorithms.

**Our Solution**: We leverage an **8-channel tensor** combining visible/infrared surface reflectance bands with physics-based spectral indices (`NDSI` and `NDVI`), processed through **Attention Gates (AGs)** that dynamically suppress background clutter and isolate true glacier margins.

---

## 🛰️ 3. The 8-Channel Multispectral Input Pipeline

Instead of relying on standard RGB images, our pipeline constructs an 8-channel tensor of shape `(Batch, 8, 256, 256)` from Copernicus Sentinel-2 Level-2A surface reflectance data:

| Channel Index | Band / Index Name | Sentinel-2 Center Wavelength | Physical Role in Cryospheric Mapping                           |
| :-----------: | :---------------: | :--------------------------: | -------------------------------------------------------------- |
| **Channel 0** |    `B2` (Blue)    |           ~490 nm            | Atmospheric scattering correction & clear water penetration    |
| **Channel 1** |   `B3` (Green)    |           ~560 nm            | Visible reflectance peak for ice; core numerator for NDSI      |
| **Channel 2** |    `B4` (Red)     |           ~665 nm            | Visible red absorption; core denominator for NDVI              |
| **Channel 3** |    `B8` (NIR)     |           ~842 nm            | High reflectance for clean ice/snow; core numerator for NDVI   |
| **Channel 4** |   `B11` (SWIR1)   |           ~1610 nm           | Strong ice absorption band; core denominator for NDSI          |
| **Channel 5** |   `B12` (SWIR2)   |           ~2190 nm           | High reflectance for rock/soil, near-zero for ice              |
| **Channel 6** |      `NDSI`       |          Calculated          | Normalized Difference Snow Index (isolates snow/ice from rock) |
| **Channel 7** |      `NDVI`       |          Calculated          | Normalized Difference Vegetation Index (isolates vegetation)   |

### Mathematical Index Formulations:

$$\text{NDSI} = \frac{\text{B3 (Green)} - \text{B11 (SWIR1)}}{\text{B3 (Green)} + \text{B11 (SWIR1)} + \epsilon}, \quad \text{NDVI} = \frac{\text{B8 (NIR)} - \text{B4 (Red)}}{\text{B8 (NIR)} + \text{B4 (Red)} + \epsilon}$$
_(where $\epsilon = 10^{-8}$ prevents division by zero. All channels are independently min-max normalized to $[0.0, 1.0]$)._

---

## 🧠 4. Attention U-Net Architecture & Attention Gates

While standard U-Nets concatenate high-resolution encoder features directly with decoder features via skip connections, they often pass redundant background noise (shadows, barren peaks).

Our **Attention U-Net** introduces **Attention Gates (AGs)** on every skip connection path:

```
[Encoder Feature (x)] ---> [ 1x1 Conv ] --+
                                           v
[Decoder Gating (g)] ---> [ 1x1 Conv ] ---> (+) ---> [ ReLU ] ---> [ 1x1 Conv + Sigmoid (ψ) ] ---> (x * ψ) ---> Concat
```

### Key Architectural Benefits:

1. **Gating Signals ($g$)**: Coarser scale feature maps from deeper decoder layers provide contextual cues about object location.
2. **Attention Coefficients ($\psi$)**: Pixel-wise weights between $0$ and $1$ are computed dynamically, multiplying directly against encoder features ($x$) to zero out irrelevant background activations before concatenation.
3. **Parameter Efficiency**: Adds minimal overhead (~31.39M parameters) while significantly improving boundary recall on thin and debris-covered glacier margins.

---

## ⚖️ 5. Loss Function & Optimization Mechanics

To handle severe class imbalance (glaciers typically occupy a small fraction of a satellite tile compared to background rock), we utilize a **Hybrid Loss Function** combining Binary Cross-Entropy and Soft Dice Loss:

$$\mathcal{L}_{\text{Hybrid}} = \alpha \cdot \mathcal{L}_{\text{BCE}}(logits, targets) + (1 - \alpha) \cdot \mathcal{L}_{\text{Dice}}(logits, targets)$$

- **BCEWithLogitsLoss**: Pixel-wise classification stability.
- **Soft Dice Loss**: Directly optimizes overlap and handles class imbalance.
- **Default Hyperparameters**: $\alpha = 0.5$, Optimizer = `AdamW` ($\text{lr} = 2\times 10^{-4}$, $\text{weight\_decay} = 10^{-4}$).

---

## 📏 6. Evaluation Metrics & Threshold Tuning

Model performance is evaluated across four core geospatial segmentation metrics:

1. **Dice Coefficient (F1-Score)**: Measures spatial overlap between predicted mask $P$ and ground-truth $G$:
   $$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G| + \epsilon}$$
2. **Intersection over Union (IoU / Jaccard Index)**:
   $$\text{IoU} = \frac{|P \cap G|}{|P \cup G| + \epsilon} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN} + \epsilon}$$
3. **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP} + \epsilon}$ (Minimizes false glacier alarms).
4. **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN} + \epsilon}$ (Ensures all glacier boundaries are detected).

---

## 📁 7. Repository Structure

```
Attention-U-Net-Glacier-Segmentation/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pip dependency manifest
├── environment.yml                # Conda environment configuration
├── .gitignore                     # Git exclusion rules
├── LICENSE                        # MIT Open-Source License
├── CITATION.cff                   # Academic citation metadata
│
├── configs/
│   └── default.yaml               # Centralized hyperparameters & paths
│
├── models/
│   ├── __init__.py                # Package module exports
│   └── attention_unet.py          # PyTorch Attention U-Net implementation
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Band normalization & NDSI/NDVI calculations
│   ├── dataset.py                 # PyTorch Dataset loader for .npy tensors
│   ├── losses.py                  # Hybrid BCE + Soft Dice Loss module
│   ├── metrics.py                 # Dice, IoU, Precision, Recall computation
│   ├── engine.py                  # Training & validation epoch iteration loops
│   └── utils.py                   # Reproducibility seeders & YAML parser
│
├── scripts/
│   ├── inspect_geotiff.py         # Geospatial raster inspection tool
│   ├── prepare_dataset.py         # Synthetic patch generator for smoke testing
│   ├── train.py                   # Main training script with checkpoint saving
│   ├── evaluate.py                # Metric evaluation & threshold analysis
│   └── predict.py                 # Full-scene sliding-window inference & export
│
├── tests/
│   ├── test_model.py              # Model shape & forward pass unit test
│   ├── test_losses.py             # Loss gradient & backward pass unit test
│   ├── test_metrics.py            # Mathematical metric verification test
│   └── test_dataset.py            # Dataset tensor loading unit test
│
├── notebooks/
│   └── glacier_segmentation_demo.ipynb # Interactive Google Colab GPU notebook
│
├── data/
│   └── README.md                  # Data placement instructions
├── outputs/
│   └── .gitkeep                   # Checkpoints & predictions tracker
└── docs/
    ├── DATASET_GUIDE.md           # Multispectral band specification guide
    ├── TRAINING_GUIDE.md          # Training instructions
    └── ARCHITECTURE.md            # Attention mechanism documentation
```

---

## ⚙️ 8. Installation & Local Setup

### Prerequisites

- Python 3.10 or higher
- Git installed on your system

### Step-by-Step Installation (Windows / Linux / macOS)

```bash
# 1. Clone your repository
git clone https://github.com/PiyushVerma/Attention-U-Net-Glacier-Segmentation.git
cd Attention-U-Net-Glacier-Segmentation

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows (PowerShell/CMD):
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# 4. Install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 9. Running Tests, Training, Evaluation & Inference

### Step A: Run Unit Tests (No Satellite Data Required)

You can verify that your software environment, model shapes, losses, and metrics are fully functional using the test suite:

```bash
# Set Python path for module resolution
$env:PYTHONPATH = "$PWD"  # On Windows PowerShell
# export PYTHONPATH=$PWD  # On Linux/macOS

# Run individual unit tests
python -m tests.test_model
python -m tests.test_losses
python -m tests.test_metrics
python -m tests.test_dataset
```

### Step B: Train the Attention U-Net

_Note: If `data/patches/` is empty, the training script automatically generates synthetic test patches for local smoke testing._

```bash
python scripts/train.py --config configs/default.yaml
```

### Step C: Evaluate the Saved Checkpoint

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

### Step D: Run Full-Scene Sliding-Window Inference & Export

```bash
python scripts/predict.py --checkpoint outputs/checkpoints/best_model.pth
```

---

## ☁️ 10. Google Colab Workflow

For training on large real-world Sentinel-2 datasets using GPU acceleration (NVIDIA T4):

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebooks/glacier_segmentation_demo.ipynb`.
3. Set hardware accelerator to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).
4. Execute the cells interactively.

---

## 📚 11. Data Attribution & References

- **Sentinel-2 Data**: Copernicus Sentinel-2 Level-2A surface reflectance data accessed via Google Earth Engine / Copernicus Open Access Hub.
- **Reference Labels**: Randolph Glacier Inventory (RGI 7.0/6.0 Region 14: South Asia West) vector polygons provided by GLIMS / NSIDC.
- **Architecture Reference**: O. Oktay et al., _"Attention U-Net: Learning Where to Look for Pancreas Segmentation"_ (arXiv:1804.03999).

---

## 📜 12. License

This project is open-source under the [MIT License](LICENSE).
