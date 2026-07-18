from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(
        ..., description="Customer gender: 'Male' or 'Female'"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Whether customer is a senior citizen (1) or not (0)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has a partner ('Yes' or 'No')"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has dependents ('Yes' or 'No')"
    )
    tenure: int = Field(
        ..., ge=0, le=120, description="Months customer has stayed with company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has phone service ('Yes' or 'No')"
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        ..., description="Whether customer has multiple lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service provider"
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has online security"
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has online backup"
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has device protection"
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has technical support"
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has streaming TV"
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has streaming movies"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract duration term"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has paperless billing"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(..., description="Customer billing payment method")
    MonthlyCharges: float = Field(
        ..., ge=0.0, description="Monthly amount billed to the customer"
    )
    TotalCharges: float = Field(
        ..., ge=0.0, description="Total cumulative amount billed to the customer"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


class PredictionOutput(BaseModel):
    churn_prediction: int = Field(
        ..., description="Binary classification (1 = Churn, 0 = Retain)"
    )
    churn_probability: float = Field(
        ..., description="Estimated churn probability score between 0.0 and 1.0"
    )
    churn_risk_level: Literal["Low", "Medium", "High"] = Field(
        ..., description="Risk category based on calibrated decision threshold"
    )
    retention_strategy: str = Field(
        ..., description="Recommended data-driven retention action"
    )


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionOutput]
    total_customers: int
    high_risk_count: int
