import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Add repository root to python search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from configs.configs import THRESHOLD
from schemas.customer import (
    CustomerInput,
    PredictionOutput,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from services.preprocessing import binary_encoder

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_prediction.pkl")

model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
        except Exception as err:
            print(f"[WARN] Failed to load model artifact at startup: {err}")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production inference microservice providing real-time churn probability scoring, "
        "risk categorization, and business retention recommendations powered by machine learning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def evaluate_risk_profile(probability: float, threshold: float):
    if probability >= 0.70:
        return (
            "High",
            "Urgent retention intervention: Trigger personalized retention discount, offer multi-year contract incentive, and assign senior account manager."
        )
    elif probability >= threshold:
        return (
            "Medium",
            "Targeted outreach: Offer tech support check-in, recommend relevant security or backup bundles, and assess service satisfaction."
        )
    else:
        return (
            "Low",
            "Loyalty maintenance: Routine engagement, standard newsletter updates, and periodic loyalty rewards."
        )


@app.get("/health", tags=["Health & Status"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "calibrated_threshold": THRESHOLD,
    }


@app.get("/model-info", tags=["Metadata"])
def get_model_info():
    return {
        "architecture": "Scikit-Learn ColumnTransformer + LogisticRegression Pipeline",
        "decision_threshold": THRESHOLD,
        "features": {
            "numeric": ["tenure", "MonthlyCharges", "TotalCharges"],
            "binary": [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "PaperlessBilling"
            ],
            "categorical": [
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaymentMethod"
            ]
        }
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_single_customer(customer: CustomerInput):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not available. Please ensure models/churn_prediction.pkl exists."
        )

    try:
        df_input = pd.DataFrame([customer.model_dump()])
        churn_prob = float(model_pipeline.predict_proba(df_input)[0, 1])
        prediction = int(churn_prob >= THRESHOLD)
        risk_level, strategy = evaluate_risk_profile(churn_prob, THRESHOLD)

        return PredictionOutput(
            churn_prediction=prediction,
            churn_probability=round(churn_prob, 4),
            churn_risk_level=risk_level,
            retention_strategy=strategy
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution failed: {str(exc)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_customers(payload: BatchPredictionRequest):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not available. Please ensure models/churn_prediction.pkl exists."
        )

    try:
        df_batch = pd.DataFrame([c.model_dump() for c in payload.customers])
        probabilities = model_pipeline.predict_proba(df_batch)[:, 1]

        outputs = []
        high_risk_count = 0

        for prob in probabilities:
            prob_float = float(prob)
            pred = int(prob_float >= THRESHOLD)
            risk_level, strategy = evaluate_risk_profile(prob_float, THRESHOLD)
            if risk_level == "High":
                high_risk_count += 1

            outputs.append(
                PredictionOutput(
                    churn_prediction=pred,
                    churn_probability=round(prob_float, 4),
                    churn_risk_level=risk_level,
                    retention_strategy=strategy
                )
            )

        return BatchPredictionResponse(
            predictions=outputs,
            total_customers=len(outputs),
            high_risk_count=high_risk_count
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch inference failed: {str(exc)}"
        )
