from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import time

app = FastAPI(title="Breast Cancer Prediction API")

# ======================
# Input Schema
# ======================
class BreastCancerInput(BaseModel):
    Clump_thickness: float
    Uniformity_of_cell_size: float
    Uniformity_of_cell_shape: float
    Marginal_adhesion: float
    Single_epithelial_cell_size: float
    Bare_nuclei: float
    Bland_chromatin: float
    Normal_nucleoli: float
    Mitoses: float


# ======================
# Dummy Model Function
# ======================
def dummy_model(X):
    """
    Simulasi model klasifikasi
    """
    score = np.mean(X)
    return 1 if score > 1 else 0


# ======================
# Inference Endpoint
# ======================
@app.post("/predict")
def predict(data: BreastCancerInput):
    start_time = time.time()

    features = np.array([[
        data.Clump_thickness,
        data.Uniformity_of_cell_size,
        data.Uniformity_of_cell_shape,
        data.Marginal_adhesion,
        data.Single_epithelial_cell_size,
        data.Bare_nuclei,
        data.Bland_chromatin,
        data.Normal_nucleoli,
        data.Mitoses
    ]])

    prediction = dummy_model(features)

    latency = time.time() - start_time

    return {
        "prediction": "malignant" if prediction == 1 else "benign",
        "latency_seconds": round(latency, 4)
    }


@app.get("/")
def health_check():
    return {"status": "API running"}
