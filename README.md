# Photo_Z_Estimation

# Photometric Redshift Estimation from Multi-Band Galaxy Images using CNNs

A deep learning-based pipeline to estimate photometric redshifts of galaxies directly from raw multi-band SDSS images using Convolutional Neural Networks (CNNs). This repository includes implementations, training notebooks, evaluation metrics, and a research paper documenting the methodology and results.

## Overview

This project applies convolutional neural networks for photometric redshift (photo-z) estimation from 32×32×5 galaxy image cubes captured in SDSS’s u, g, r, i, z photometric bands. The model performs regression to predict redshift values using end-to-end deep learning.

## Repository Structure
```
photometric-redshift-cnn/
├── sd2.ipynb              # CNN training notebook using 2σ clipped redshift dataset
├── whole.ipynb            # CNN training notebook on full merged dataset
├── Photoz.pdf  # Research paper detailing methodology and results
├── modelSD2.h5            # Trained model on clipped dataset
├── redshift_model.h5      # Trained model on full dataset
└── README.md              # This file
```
## Dataset

- **Source:** Sloan Digital Sky Survey (SDSS) DR12 via NERSC  
- **Number of Images:** ~400,000  
- **Image Shape:** 32×32 pixels with 5 channels (ugriz)  
- **Redshift Range:** 0.0 – 0.40  
- **Format:** `.h5` with `images`, `specz_redshift`, and `specz_redshift_err` datasets  

## Model Architecture

A CNN built using TensorFlow/Keras:

- Two Conv2D layers with 32 and 64 filters respectively  
- MaxPooling after each convolutional block  
- Flatten layer  
- Fully connected layers with 220 and 64 units  
- Dropout layers (rate = 0.5) for regularization  
- Output layer with sigmoid activation to predict normalized redshift  

Total parameters: ~421,000


## Results

| Metric                    | Full Dataset     | Clipped Dataset (±2σ) |
|---------------------------|------------------|------------------------|
| Mean Absolute Error (MAE) | 0.0304           | 0.0556                 |
| Root Mean Squared Error   | 0.0412           | 0.0751                 |
| R² Score                  | 0.9041           | 0.8825                 |
| Bias                      | 0.00029          | -0.00067               |
| Precision (1.48×Δz)       | 0.0269           | 0.0427                 |
| Δz > 0.05                 | 10.86%           | 27.32%                 |
| Δz > 0.10                 | 0.91%            | 5.90%                  |
| Δz > 0.15                 | 0.12%            | 1.28%                  |

**Key Insight:** The model trained on the full dataset (with outliers) performs significantly better than the one trained on the clipped dataset.

## How to Run

### Requirements

- Python 3.8+  
- TensorFlow 2.x  
- NumPy  
- Pandas  
- h5py  
- scikit-learn  
- matplotlib  
- Jupyter Notebook  

### Training

To train using the 2σ clipped dataset, open and run the notebook:

```
jupyter notebook sd2.ipynb
```

To train using the full dataset, open and run the notebook:

```
jupyter notebook whole.ipynb
```

Both notebooks will save trained models as .h5 files.

Paper
Title: Photometric Redshift Estimation from Multi-Band Galaxy Images using CNNs

Authors: Atharva Surve, Prajot Patil, Ariyan Muddapur, Anshuman Padhi, Pramod H. Kachare

Affiliation: Ramrao Adik Institute of Technology, D.Y. Patil Deemed University, Navi Mumbai

PDF: See AIMLGroup22_Paper.pdf

Future Work
Cross-survey generalization
Band importance analysis
Faster inference with lightweight models
Real-time predictions and ROI detection


Citation
If you use this work in your research, please cite:
```
@article{Patil2025RedshiftCNN,
  title={Photometric Redshift Estimation from Multi-Band Galaxy Images using CNNs},
  author={Prajot Patil et al.},
  journal={Unpublished},
  year={2025}
}
```

