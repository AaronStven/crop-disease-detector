# 🌱 AI-Based Crop Disease Detection and Advisory Web Platform (Ongoing Thesis)

An undergraduate thesis project (B.Eng Computer Science) focusing on **Rice, Cassava, and Maize in Madagascar**. This repository tracks the ongoing development of the PyTorch model training pipeline and the FastAPI backend for the web platform.

> **🚧 Project Status: Phase 1 Complete (Backend & Model Architecture)**
> The core API and model definitions are functional. Current work is focused on dataset expansion, training the models, and developing the React frontend.

## 📖 Thesis Abstract
Agriculture is central to Madagascar's economy, but crop diseases significantly constrain productivity. This study investigates the comparative performance of a **Custom CNN** and **Transfer Learning models (ResNet and MobileNet)** for crop disease classification. The selected model will be integrated into a web-based advisory platform using FastAPI. 
👉 **[Read the full 1-page thesis summary here](docs/thesis_summary.md)**

## ✅ Completed (Phase 1)
- **Backend API:** FastAPI server successfully accepts image uploads and returns mock predictions.
- **Model Architecture:** PyTorch scripts defined for Custom CNN, ResNet18, and MobileNetV2.
- **Thesis Documentation:** Abstract and methodology summarized in the `docs/` folder.
- **Data Pipeline:** Augmentation and preprocessing transforms initialized.

## 🚧 Ongoing Work (Phase 2 & 3)
- [ ] **Dataset Expansion:** Acquiring and cleaning a larger dataset of Madagascar-specific Rice, Cassava, and Maize leaf images.
- [ ] **Model Training:** Running training loops to compare Custom CNN vs. ResNet vs. MobileNet on Accuracy and F1-Score.
- [ ] **Explainable AI (XAI):** Integrating Grad-CAM to generate visual heatmaps of detected diseases.
- [ ] **Frontend Development:** Building the React web interface for farmers to upload images and view advisory information.

## 🛠 Tech Stack
- **Deep Learning:** PyTorch, Torchvision
- **Backend:** FastAPI, Uvicorn
- **Data Processing:** PIL, NumPy
- **Frontend (Planned):** React

## 🚀 How to Run the Backend (Phase 1)
1. Clone the repo and navigate into it.
2. Create a virtual environment and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the API: `uvicorn api.main:app --reload`
5. Open `http://127.0.0.1:8000/docs` to test the image upload endpoint via UI.