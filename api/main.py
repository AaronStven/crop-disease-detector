from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import random

app = FastAPI(
    title="Crop Disease Detector API", 
    description="API for Rice, Cassava, and Maize disease detection (Thesis Project)"
)

# Classes specific to the thesis
CLASSES = [
    "Rice - Healthy", "Rice - Bacterial Blight", "Rice - Blast", "Rice - Brown Spot",
    "Cassava - Healthy", "Cassava - Mosaic Disease", "Cassava - Brown Leaf Spot",
    "Maize - Healthy", "Maize - Common Rust", "Maize - Northern Leaf Blight"
]

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Crop Disease Detector for Madagascar"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Simulate PyTorch model inference
    predicted_class = random.choice(CLASSES)
    confidence = round(random.uniform(0.85, 0.99), 3)
    
    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": confidence,
        "explainability_heatmap_url": f"/static/heatmaps/{file.filename}_gradcam.jpg",
        "advisory": "Please consult local agricultural extensions. Ensure proper crop rotation and apply recommended fungicides."
    }