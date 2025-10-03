# Exoplanet Detection with Deep Learning - MVP Pipeline

## Overview
This notebook implements an end-to-end machine learning pipeline for detecting exoplanets from telescope light curve data. It uses a dual-branch Convolutional Neural Network (CNN) architecture inspired by AstroNet to classify stellar light curves as containing planetary transits or not. The pipeline includes data exploration, preprocessing, model training, and transfer learning experiments.

## Key Features:
Multi-catalog data exploration (K2, TESS ExoMiner, TOI, Kepler Cumulative)
Light curve preprocessing with detrending and normalization
Dual-view CNN architecture (global + local views)
Transfer learning demonstration for domain adaptation
Comprehensive evaluation metrics and visualizations

## Requirements
Python Version: 3.7+
Core Libraries:
torch>=1.9.0
torchvision
torchaudio
lightkurve
astropy
numpy
scipy
scikit-learn
matplotlib
seaborn
tensorflow>=2.6.0
pandas

## Installation
[
Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Install astronomy and ML libraries
pip install lightkurve astropy numpy scipy scikit-learn matplotlib seaborn tensorflow pandas

Clone ExoMiner repository
git clone https://github.com/google-research/exoplanet-ml.git
]

# Dataset Description
The notebook works with four exoplanet catalogs:

1- K2 Planets and Candidates (k2pandc_*.csv) - K2 mission confirmed planets and candidates
2- ExoMiner (exominer_vetting_*.csv) - TESS machine learning candidate scores
3- TOI (TESS Objects of Interest) (TOI_*.csv) - TESS candidate list
4- Cumulative Kepler (cumulative_*.csv) - Kepler mission cumulative catalog

**Expected CSV Format:** Catalogs contain metadata with columns like:

Target IDs (TIC, EPIC, KepID)
Disposition status (CONFIRMED, CANDIDATE, FALSE POSITIVE)
Orbital parameters (period, depth, duration)

# Notebook Structure
## 1. Data Loading & Exploration
Handles CSV files with comment headers (skips metadata rows)
Parses multiple catalog formats
Extracts unique target identifiers for each mission

## 2. Label Creation
Creates binary classification labels (planet vs non-planet)
Uses disposition columns or confidence scores
Handles class imbalance with proper stratification

## 3. Light Curve Download (Optional)
Uses lightkurve to download TESS/Kepler light curves
Downloads from MAST (Mikulski Archive for Space Telescopes)
Saves light curves as compressed NumPy arrays

**4. Synthetic Data Generation**
Generates realistic synthetic light curves for testing
Models stellar variability, noise, and transit signals
Configurable parameters: period, depth, duration

## 5. Preprocessing Pipeline
**Detrending:** Savitzky-Golay filter removes long-term trends
**Normalization:** Robust scaling using median and MAD
**Global View:** 2000-point resampled full light curve
**Local View:** 201-point zoomed window around transit
Outputs fixed-size arrays for CNN input

## 6. CNN Model Architecture
**Dual-Branch Design:**

**Global Branch:** Captures long-term patterns

Conv1D (16 filters, kernel=5) → MaxPool → Conv1D → MaxPool → Dense


**Local Branch:** Captures transit details

Conv1D (16 filters, kernel=5) → MaxPool → Conv1D → MaxPool → Dense


**Fusion:** Concatenate features → Dense layers → Sigmoid output

## 7. Model Training
70/15/15 train/validation/test split
Class weighting for imbalanced data
Callbacks: Early stopping, model checkpointing, learning rate reduction
Metrics: Accuracy, Precision, Recall, AUC

## 8. Evaluation
Confusion matrix visualization
ROC curve with AUC score
Classification report with per-class metrics
Training history plots (loss, accuracy, precision, recall)

## 9. Transfer Learning
**Three Strategies Compared:**

**Baseline:** No transfer (poor on new data)
**Frozen Layers:** Freeze early layers, train final layers
**Full Fine-tuning:** Unfreeze all layers with low learning rate
**From Scratch:** Train new model entirely

# Example Outputs
**Files Created**

preprocessed_data.npz              # Processed light curves (global + local views)
astronet_best_model.h5             # Best trained baseline model
astronet_transfer_learned.h5       # Transfer learned model
training_history.csv               # Epoch-by-epoch training metrics
test_predictions.csv               # Test set predictions with probabilities
model_metrics.csv                  # Summary statistics
confusion_matrix.png               # Confusion matrix heatmap
roc_curve.png                      # ROC curve plot
training_history.png               # Training curves (4-panel)
transfer_learning_comparison.png   # Transfer learning results

## Performance Metrics (Example)
Test Accuracy:  0.9333
Test Precision: 0.9200
Test Recall:    0.9400
Test AUC:       0.9750

## Visualizations
Sample raw light curves (planet vs non-planet)
Preprocessed light curves (global + local views)
Training convergence curves
Confusion matrices
ROC curves
Transfer learning comparison bar charts

## Notes & Limitations
**Current Limitations:**

**Synthetic Data:** Uses generated light curves instead of real TESS/Kepler data for demonstration
**Small Sample Size:** 100-200 samples in examples (production needs 10,000+)
**Simple Architecture:** Simplified AstroNet (production uses deeper networks)
**No Uncertainty:** Lacks Bayesian inference or confidence intervals
**Manual Transit Detection:** Local view uses simple threshold-based detection

## Known Issues:

CSV parsing requires handling various comment line formats
Light curve download can fail due to network/MAST issues
Class imbalance requires careful sampling strategies
Transfer learning benefits vary with domain shift magnitude

## Best Practices & Lessons Learned

**1- Data Quality First**
Always validate CSV parsing (many astronomy catalogs have metadata rows)
Check for NaN values and remove before normalization
Visualize raw data before preprocessing


**2-Preprocessing Robustness**
Use median/MAD instead of mean/std (outlier-resistant)
Detrend before normalization
Fixed-length arrays simplify model architecture


**3-Model Training**
Use class weights for imbalanced datasets
Monitor multiple metrics (not just accuracy)
Save best model based on validation performance
Implement early stopping to avoid overfitting


**4-Transfer Learning Strategy**
Start with frozen layers, then gradually unfreeze
Use lower learning rates for fine-tuning (0.0001 vs 0.001)
Compare against from-scratch training as baseline
Transfer learning works best with 10-20% of original dataset size


**5-Production Considerations**
Scale to full catalogs (10,000+ targets)
Implement batched light curve downloads
Add GPU support for faster training
Deploy as REST API for real-time inference
Version control datasets and models



# Next Steps for Production
Download real light curves from MAST using lightkurve
Expand dataset to 10,000+ labeled examples
Implement data augmentation (phase shifts, noise injection)
Add ensemble models for improved robustness
Integrate uncertainty quantification (e.g., MC Dropout)
Create web dashboard for predictions visualization
Set up automated retraining pipeline
Deploy model to cloud infrastructure
