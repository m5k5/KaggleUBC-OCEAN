# UBC-OCEAN: Ovarian Histology Classification Pipeline

## Overview

This repository implements an end-to-end Deep Learning pipeline for Digital Pathology, specifically targeting the UBC-OCEAN challenge: classifying ovarian tumors from histology slides. The project bridges Medical Physics (imaging physics, WSI handling), Machine Learning (CNNs, Multi-Instance Learning), and Software Engineering.

## Methodology & Architecture

The solution follows a Multi-Stage Inference Strategy designed to balance accuracy with computational constraints:

1. Data Engineering (py_wsi)
   - Custom wrapper for Whole Slide Images (WSI) using OpenSlide.
   - Automated patch extraction (LMDB/HDF5 storage) with boundary detection and label mapping.
   - Domain-specific augmentation pipelines (color jitter, flips, normalization).

1. Model Design
   - Feature Extraction: Utilization of pre-trained backbones (EdgeNeXt, ResNet34d, CAFormer) optimized for histology textures.
   - Multi-Instance Learning (MIL): Aggregation strategies (max-pooling, mean-pooling) to translate patch-level predictions to slide-level diagnosis.
   - Optimization: Custom loss functions (Focal Loss, Balanced Accuracy) to address class imbalance (e.g., HGSC vs. LGSC).
1. Deployment
   - TorchScript: Models exported for static inference graphs.
   - TensorBoard: Comprehensive logging for hyperparameter analysis.

## Key Achievements

- Systems Thinking: Built a scalable extraction pipeline handling GBs of histology data efficiently.
- Domain Adaptation: Integrated histopathological constraints (e.g., tumor types: LGSC, HGSC, CC) directly into the classification logic.
- Optimization: Successfully exported models to TorchScript for optimized deployment environments.
- First Principles Approach: Analyzed class distributions to implement focal loss and balanced metrics, ensuring robustness against rare classes.
  Quick Start

## Theoretical Background

- Physics & Modeling: First-principles understanding of imaging and signal processing in pathological data.
- AI/ML: Deep Learning architectures, loss engineering, and transfer learning.
- Full System: From raw data ingestion to inference artifacts.
- Communication: Experimental tracking and reproducibility via structured logs.
