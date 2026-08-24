# Biometric Facial Recognition System Using CNN & Ant Colony Optimization (ACO)

An end-to-end biometric facial recognition pipeline developed in **MATLAB** using the **CelebA** dataset. The project integrates Deep Learning (CNN) for feature extraction with **Ant Colony Optimization (ACO)** for feature selection, drastically reducing dimensionality while maintaining high classification performance.

---

## 📌 Project Overview
* **Feature Extraction:** Pre-trained Convolutional Neural Network (CNN) backbone extracting latent visual features.
* **Feature Optimization:** Ant Colony Optimization (ACO) algorithm to filter out redundant attributes and select optimal features.
* **Dataset:** Evaluated on the benchmark **CelebA** dataset.
* **User Interface:** MATLAB App Designer GUI (`app2.mlapp`) for real-time testing and inference.
* **Performance:** Pre-trained models included, achieving up to **91% accuracy** (`Best_Model_91.mat`).

---

## 📁 Repository Structure
* `CelebA/` : Benchmark dataset directory.
* `MATLAB-GP.prj` : MATLAB Project configuration file.
* `main.m` : Master script to run the full workflow.
* `prepareDataset.m` : Data loading, labeling, and partitioning.
* `preprocessImages.m` : Image normalization and resizing.
* `trainACO_CNN.m` : CNN feature extraction and ACO optimization.
* `evaluateModel.m` & `testModel.m` : Evaluation metrics, confusion matrix, and accuracy testing.
* `app2.mlapp` : Interactive GUI interface.
* `*.mat` files : Pre-trained model checkpoints (e.g., `Best_Model_91.mat`, `FinalModel.mat`).

---

## 🛠 Requirements & Toolboxes
* MATLAB (R2023a or newer recommended)
* Deep Learning Toolbox
* Statistics and Machine Learning Toolbox
* Computer Vision Toolbox
* App Designer

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
