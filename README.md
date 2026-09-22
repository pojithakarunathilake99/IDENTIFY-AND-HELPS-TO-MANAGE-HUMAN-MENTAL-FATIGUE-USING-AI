# Identify and Help Manage Human Mental Fatigue Using AI

An MSc thesis project applying machine learning and deep learning to detect **mental fatigue and stress states** from wearable physiological sensor data — without relying on EEG. The system is built on two public, non-EEG physiological datasets and includes a real-time inference pipeline that pushes predictions to Firebase for consumption by a companion app/device.

> Based on the MSc Artificial Intelligence thesis "Identify and Helps to Manage Human Mental Fatigue Using AI", Birmingham City University.
> Advisors: Dr. Hansi Hettiarachchi and Dr. Mariam Adedoyin-Olowe.
> DOI: [10.13140/RG.2.2.22829.87521](https://doi.org/10.13140/RG.2.2.22829.87521)

---

## Overview

Traditional fatigue/stress detection relies on EEG, which is accurate but impractical outside a lab (expensive equipment, skin-contact electrodes, low wearability). This project instead asks: can we reliably classify mental stress and fatigue states using only signals a consumer wearable can already capture — heart rate, blood oxygen saturation (SpO2), chest acceleration, respiration, and skin temperature?

Two complementary datasets are used to explore this:

| Dataset | Signals used | Classes |
|---|---|---|
| **WESAD** (Wearable Stress and Affect Detection) | Chest-worn accelerometer (x/y/z), ECG, respiration, temperature | Baseline / Stress / Amusement / Meditation |
| **Non-EEG Dataset for Assessment of Neurological Status** | SpO2, heart rate (per-second aggregated statistics) | Relax / Physical Stress / Emotional Stress / Cognitive Stress |

For each dataset, the pipeline goes from raw sensor files to cleaned/merged data to engineered features to classical ML and deep learning models to a real-time prediction service.

---

## Repository Structure

```
├── wesad/                          # WESAD dataset pipeline
│   ├── 1 pkl to csv.ipynb          # Convert WESAD's native .pkl subject files to CSV
│   ├── 2 combined.ipynb            # Merge all subjects into a single dataset
│   ├── 3 features.ipynb            # Feature extraction / selection
│   ├── 4 DL .ipynb                 # Deep learning model (Keras, hyperparameter search)
│   ├── 5 firebase.py               # Real-time inference: reads sensor data from Firebase,
│   │                                  runs the trained model, writes predictions back
│   └── 94precent.h5                # Trained Keras model checkpoint
│
├── NON-EEG/                        # Non-EEG (SpO2/HR) dataset pipeline
│   ├── 0 get_spo2_files.ipynb      # Load raw per-subject SpO2/HR recordings
│   ├── 1 Merged.ipynb              # Merge subjects/conditions into one dataset
│   ├── 2 data_cleaning.ipynb       # Cleaning and preprocessing
│   ├── 3 correlation.ipynb         # Feature correlation analysis
│   ├── 4 svm.ipynb                 # Support Vector Machine classifier
│   ├── 5 svm Hyperparameter.ipynb  # SVM hyperparameter tuning
│   └── 6 ml models.ipynb           # Comparison of additional ML models
│
└── README.md
```

Notebooks in each folder are numbered in the order they should be run — each stage's output feeds the next.

---

## Methodology

1. **Data preparation** – Raw sensor recordings (WESAD `.pkl` files; Non-EEG per-subject SpO2/HR logs) are parsed, merged across subjects, and cleaned.
2. **Feature engineering** – For the Non-EEG dataset, per-second statistical features (mean, median, standard deviation, min, max) are computed for SpO2 and heart rate. For WESAD, chest sensor channels (accelerometer, ECG, respiration, temperature) are used directly/aggregated per window.
3. **Exploratory analysis** – Correlation analysis to understand which signals are most predictive of each stress/fatigue state.
4. **Classical machine learning** – A Support Vector Machine (SVM) is trained on the Non-EEG features, with a dedicated hyperparameter tuning pass and comparison against other ML models.
5. **Deep learning** – A feed-forward neural network (Keras/TensorFlow) is trained on the WESAD features, with architecture and hyperparameters (layer sizes, dropout rate, learning rate) selected via Keras Tuner (Bayesian optimisation).
6. **Real-time deployment** – `wesad/5 firebase.py` loads the trained model and continuously polls a Firebase Realtime Database for live accelerometer readings (e.g. from a connected wearable/mobile app), runs inference, and writes the predicted state back to Firebase for the client app to display.

---

## Tech Stack

- **Language:** Python (Jupyter Notebooks)
- **Data processing:** pandas, NumPy
- **Classical ML:** scikit-learn (SVM, StandardScaler, train/test pipelines)
- **Deep learning:** TensorFlow / Keras, Keras Tuner (Bayesian & Random Search)
- **Deployment:** Firebase Admin SDK (Realtime Database) for live sensor ingestion and prediction serving

---

## Datasets

- **WESAD** — Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection.
- **Non-EEG Dataset for Assessment of Neurological Status** — Birjandtalab, J., Cogan, D., Pouyan, M.B., & Nourani, M. (2016), available via PhysioNet.

Both datasets are publicly available; they are not redistributed in this repository and must be downloaded separately from their original sources.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/pojithakarunathilake99/IDENTIFY-AND-HELPS-TO-MANAGE-HUMAN-MENTAL-FATIGUE-USING-AI.git
cd IDENTIFY-AND-HELPS-TO-MANAGE-HUMAN-MENTAL-FATIGUE-USING-AI

# Install dependencies
pip install pandas numpy scikit-learn tensorflow keras-tuner firebase-admin jupyter

# Launch the notebooks (run in numbered order within each folder)
jupyter notebook
```

You will need your own copies of the WESAD and Non-EEG datasets, and (for the real-time component) a Firebase project with a Realtime Database and a service account credentials file.

---

## Author

**Pojitha Karunathilake**
MSc Artificial Intelligence (Distinction), Birmingham City University
[GitHub](https://github.com/pojithakarunathilake99) · [pojithatech.co.uk](https://pojithatech.co.uk)

---

## Citation

If you use this work, please cite:

```
Hettiarachchilalage, P. (2023). Identify and Helps to Manage Human Mental Fatigue Using AI.
MSc Thesis, Birmingham City University. DOI: 10.13140/RG.2.2.22829.87521
```
