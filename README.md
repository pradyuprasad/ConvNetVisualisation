# CNN Feature Visualization

A project to visualize and analyze learned features in a CNN trained on the STL-10 dataset, implementing techniques from "Visualizing and Understanding Convolutional Networks" (Zeiler & Fergus, 2014).

## Overview

This project implements a 4-layer CNN for image classification and provides tools to visualize what features each filter learns through deconvolution. It includes:
- CNN training and evaluation
- Deconvolutional network for feature visualization
- Filter activation analysis
- Heatmap generation of filter responses

## Setup

1. Create and activate a virtual environment:
```
uv sync
```

## Usage

All commands should be run with uv run:

1. Train the CNN:
```
uv run train_CNN.py
```

2. Generate deconvolution visualizations:
```
uv run deconv_full.py
```

3. Analyze filter activations:
```
uv run analyse_projections.py
```
4. Generate heatmaps:
```
uv run heatmap.py
```

## Project Structure

- CNN.py: Model architecture definition
- train_CNN.py: Training loop and model evaluation
- deconv.py: Basic deconvolution implementation
- deconv_full.py: Full deconvolution network
- analyse_projections.py: Filter activation analysis
- heatmap.py: Visualization generation
- load_data.py: Data loading utilities

## Results

The project generates:
- Filter activation heatmaps per class
- Statistical analysis of filter specialization
- Visualizations of what features each filter learns

## Implementation Notes

1. The CNN architecture uses:
   - 4 convolutional layers
   - Batch normalization
   - ReLU activation
   - Max pooling
   - Dropout for regularization

2. Deconvolution process:
   - Reuses weights from trained CNN
   - Handles pooling indices properly
   - Generates visualizations for all 256 filters

3. Analysis includes:
   - Non-zero activation counting
   - Mean activation values
   - Variance analysis
   - Per-class filter specialization

## Acknowledgments

Based on techniques from:
Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." European conference on computer vision. Springer, Cham, 2014.
