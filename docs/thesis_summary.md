# Thesis Overview: AI-Based Crop Disease Detection in Madagascar

## The Problem
Agriculture is central to Madagascar's economy, but crop diseases (specifically in Rice, Cassava, and Maize) significantly reduce yields. Farmers often lack access to agricultural experts, leading to delayed or inaccurate visual diagnoses.

## The Research
This study compares the performance of a **Custom CNN** against **Transfer Learning models (ResNet and MobileNet)** for image-based disease classification. 

## Methodology
- **Dataset:** Sourced from public agricultural datasets, focused on Rice, Cassava, and Maize. Split into Train/Val/Test.
- **Models:** 
  1. Custom CNN (Baseline)
  2. ResNet (High capacity)
  3. MobileNet (Computational efficiency)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Inference Time.

## System Architecture
The best-performing model is integrated into a web platform using **PyTorch** for inference and **FastAPI** for the backend. The system allows users to upload a leaf image and receive a disease prediction alongside advisory information.