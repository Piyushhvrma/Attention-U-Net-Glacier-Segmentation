# 🧊 Attention U-Net Multispectral Glacier Segmentation Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform: Windows | Linux | Colab](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Colab-green.svg)]()

An independent, production-grade Deep Learning framework implementing an **8-Channel Attention U-Net** enhanced with **CBAM (Convolutional Block Attention Modules)** and **Focal Tversky Loss** for automated, high-precision glacier and glacial lake segmentation in rugged alpine terrain using Copernicus Sentinel-2 multispectral imagery and Randolph Glacier Inventory (RGI) reference labels.

- **Academic Context**: B.Tech Information Technology (7th Semester Minor Project), National Institute of Technology (NIT) Raipur.
- **Authors**: Piyush Verma, Deepanshu Sahu, Rameshwar Chandravanshi.

---

## 📑 Table of Contents

1. [Project Overview & Motivation](#-project-overview--motivation)
2. [Problem Statement & Scientific Challenges](#-problem-statement--scientific-challenges)
3. [Version Evolution & Changelog (v1.0 to v2.0)](#-version-evolution--changelog-v10-to-v20)
4. [The 8-Channel Multispectral Input Pipeline](#-the-8-channel-multispectral-input-pipeline)
5. [Attention U-Net Architecture & CBAM Integration](#-attention-u-net-architecture--cbam-integration)
6. [Advanced Loss Functions (Hybrid BCE-Dice & Focal Tversky)](#-advanced-loss-functions-hybrid-bce-dice--focal-tversky)
7. [Statistical Sensor & Dataset Analysis Framework](#-statistical-sensor--dataset-analysis-framework)
8. [Evaluation Metrics & Threshold Tuning](#-evaluation-metrics--threshold-tuning)
9. [Repository Structure](#-repository-structure)
10. [Installation & Local Setup](#-installation--local-setup)
11. [Running Tests, Training, Evaluation & Inference](#-running-tests-training-evaluation--inference)
12. [Google Colab Workflow](#-google-colab-workflow)
13. [Data Attribution & References](#-data-attribution--references)
14. [License](#-license)

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

**Our Solution**: We leverage an **8-channel tensor** combining visible/infrared surface reflectance bands with physics-based spectral indices (`NDSI` and `NDVI`), processed through **Attention Gates (AGs)** and **CBAM** that dynamically suppress background clutter and isolate true glacier margins.

---

## 📈 3. Version Evolution & Changelog (v1.0 to v2.0)

### Version 2.0 (Current - Advanced Upgrades & Statistical Rigor)

- **CBAM (Convolutional Block Attention Module)**: Upgraded standard spatial-only Attention Gates into dual **Channel and Spatial Attention** modules. Channel attention dynamically weights _what_ spectral bands matter (SWIR, NIR, NDSI, NDVI), while spatial attention isolates _where_ glacier boundaries lie.
- **Advanced Loss Formulation**: Introduced **Focal Tversky Loss** alongside Hybrid BCE-Dice Loss to heavily penalize false negatives and successfully capture elusive debris-covered glacier ice.
- **Statistical Sensor Analysis Framework**: Added `scripts/statistical_sensor_analysis.py` to execute Mann-Whitney U tests, Wilcoxon signed-rank tests, descriptive statistics, and bootstrapped 95% confidence intervals across sensor/dataset distributions.
- **Stability & Refactoring**: Resolved tensor channel dimension alignments, added automatic fallback synthetic dataset generation, and optimized execution across CPU and CUDA backends.

### Version 1.0 (Initial Baseline Implementation)

- **Core Architecture**: Developed base PyTorch implementation of standard Attention U-Net (`models/attention_unet.py`).
- **Data Ingestion**: Built 8-channel multispectral patch loaders (`src/dataset.py`) and preprocessing routines (`src/preprocessing.py`).
- **Training & Evaluation**: Established training loops, learning rate schedulers, and metric evaluators tracking Dice, IoU, Precision, and Recall.

---

## 🛰️ 4. The 8-Channel Multispectral Input Pipeline

Our pipeline constructs an 8-channel tensor of shape `(Batch, 8, 256, 256)` from Copernicus Sentinel-2 Level-2A surface reflectance data:

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
_(where $\epsilon = 10^{-8}$ prevents division by zero. All channels are min-max normalized to $[0.0, 1.0]$)._

---

## 🧠 5. Attention U-Net Architecture & CBAM Integration

Standard U-Nets concatenate encoder features with decoder features via skip connections, often passing redundant background noise. Our upgraded **Attention U-Net** integrates **CBAM** into the attention gates on every skip connection:

1. **Channel Attention**: Emphasizes informative spectral channels (e.g., SWIR bands for debris ice).
2. **Spatial Attention**: Focuss on 'where' the glacier boundary is located.
3. **Gating Signals**: Coarser scale feature maps from deeper layers provide contextual guidance.

---

## ⚖️ 6. Advanced Loss Functions (Hybrid BCE-Dice & Focal Tversky)

To handle severe class imbalance, we implement two advanced loss modules in `src/losses.py`:

$$\mathcal{L}_{\text{Hybrid}} = \alpha \cdot \mathcal{L}_{\text{BCE}}(logits, targets) + (1 - \alpha) \cdot \mathcal{L}_{\text{Dice}}(logits, targets)$$

$$\mathcal{L}_{\text{FocalTversky}} = \left(1 - \frac{\text{TP} + \text{s}}{\text{TP} + \alpha \cdot \text{FP} + \beta \cdot \text{FN} + \text{s}}\right)^{\frac{1}{\gamma}}$$
_(where weighting parameters $\alpha = 0.3$, $\beta = 0.7$, and $\gamma = 4/3$ heavily penalize false negatives to capture debris-covered ice margins)._

---

## 📊 7. Statistical Sensor & Dataset Analysis Framework

To rigorously answer comparative research questions between sensor datasets (e.g., Sentinel-2 vs alternate satellite products), our framework implements statistical hypothesis tests via `scripts/statistical_sensor_analysis.py`:

- **Mann-Whitney U Test**: Evaluates independent sample performance distributions.
- **Wilcoxon Signed-Rank Test**: Assesses paired distribution differences.
- **Bootstrapped 95% Confidence Intervals**: Quantifies statistical certainty of mean Dice improvements.

---

## 📏 8. Evaluation Metrics & Threshold Tuning

Model performance is evaluated across four core geospatial segmentation metrics:

1. **Dice Coefficient (F1-Score)**: $\frac{2 |P \cap G|}{|P| + |G| + \epsilon}$
2. **Intersection over Union (IoU)**: $\frac{|P \cap G|}{|P \cup G| + \epsilon}$
3. **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP} + \epsilon}$
4. **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN} + \epsilon}$

---

## 📁 9. Repository Structure

```
Attention-U-Net-Glacier-Segmentation/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pip dependency manifest
├── .gitignore                     # Git exclusion rules
├── LICENSE                        # MIT Open-Source License
│
├── configs/
│   └── default.yaml               # Centralized hyperparameters & paths
│
├── models/
│   └── attention_unet.py          # Attention U-Net with CBAM modules
│
├── src/
│   ├── dataset.py                 # PyTorch Dataset loader for .npy tensors
│   ├── engine.py                  # Training & validation epoch loops
│   ├── losses.py                  # Hybrid BCE-Dice & Focal Tversky Loss
│   ├── metrics.py                 # Dice, IoU, Precision, Recall computation
│   ├── preprocessing.py           # Band normalization & NDSI/NDVI calculators
│   └── utils.py                   # Reproducibility seeders & YAML parser
│
├── scripts/
│   ├── evaluate.py                # Metric evaluation & threshold analysis
│   ├── inspect_geotiff.py         # Geospatial raster inspection tool
│   ├── predict.py                 # Full-scene sliding-window inference & export
│   ├── prepare_dataset.py         # Synthetic patch generator for smoke testing
│   ├── statistical_sensor_analysis.py # Mann-Whitney U & Wilcoxon tests
│   └── train.py                   # Main training script with checkpoint saving
│
├── tests/
│   ├── test_dataset.py            # Unit test for data loader
│   ├── test_losses.py             # Unit test for loss functions
│   ├── test_metrics.py            # Unit test for mathematical metrics
│   └── test_model.py              # Unit test for model forward pass
│
├── notebooks/
│   └── glacier_segmentation_demo.ipynb # Interactive Google Colab GPU notebook
│
├── outputs/
│   ├── checkpoints/               # Saved model weight checkpoints (.pth)
│   └── predictions/               # Predicted masks and visual overlays (.png, .npy)
└── docs/
    ├── ARCHITECTURE.md            # Detailed architectural breakdown
    ├── DATASET_GUIDE.md           # Multispectral band specification guide
    └── TRAINING_GUIDE.md          # Training instructions
```

---

## ⚙️ 10. Installation & Local Setup

```bash
# 1. Clone your repository
git clone https://github.com/PiyushVerma/Attention-U-Net-Glacier-Segmentation.git
cd Attention-U-Net-Glacier-Segmentation

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # On Windows PowerShell

# 3. Install required packages
pip install --upgrade pip
pip install torch torchvision rasterio numpy matplotlib pyyaml scipy scikit-learn
```

---

## 🚀 11. Running Tests, Training, Evaluation & Inference

Set your Python path in PowerShell so modules resolve correctly:

```powershell
$env:PYTHONPATH = "$PWD"
```

1. **Run Statistical Sensor Analysis:**
   ```powershell
   python scripts/statistical_sensor_analysis.py
   ```
2. **Train the Attention U-Net:**
   _(Note: If `data/patches/` is empty, synthetic test patches are automatically generated)._
   ```powershell
   python scripts/train.py --config configs/default.yaml
   ```
3. **Evaluate the Saved Checkpoint:**
   ```powershell
   python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
   ```
4. **Run Full-Scene Inference & Visualization:**
   ```powershell
   python scripts/predict.py --checkpoint outputs/checkpoints/best_model.pth
   ```

---

## ☁️ 12. Google Colab Workflow

Upload `notebooks/glacier_segmentation_demo.ipynb` to Google Colab, select a **T4 GPU** runtime, and execute cells interactively.

---

## 📚 13. Data Attribution & References

- **Sentinel-2 Data**: Copernicus Sentinel-2 Level-2A surface reflectance data accessed via Google Earth Engine / Copernicus Open Access Hub.
- **Reference Labels**: Randolph Glacier Inventory (RGI 7.0/6.0 Region 14: South Asia West) vector polygons provided by GLIMS / NSIDC.
- **Architecture Reference**: O. Oktay et al., _“Attention U-Net: Learning Where to Look for Pancreas Segmentation”_ (arXiv:1804.03999).

---

## 📜 14. License

This project is open-source under the [MIT License](LICENSE).
