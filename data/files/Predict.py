import numpy as np
import pandas as pd
import joblib

# load
loaded = joblib.load("bestt_model.pkl")
model = loaded["model"]

cols = joblib.load("columns.pkl")

#new data
data=pd.DataFrame([{
    "Gender":"Female",  
    "Age":25,
    "SeniorCitizen":0,
    "Partner":"No",
    "Dependents":"Yes",
    "Tenure": 12,
    "PhoneService":"Yes",
    "MultipleLines":"Yes",
    "OnlineSecurity":"Yes",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies":"No",
    "PaperlessBilling":"Yes",
    "MonthlyCharges":50.65,
    "TotalCharges": 550,
    "InternetService":"DSL",
    "Contract":"One year",
    "PaymentMethod":"Mailed check"
}])

# prediction
labels = {0: "No", 1: "Yes"}
data["SeniorCitizen"] = data["SeniorCitizen"].map({"Yes": 1, "No": 0})

prediction = model.predict(data)
print("Churn Prediction:", labels[prediction[0]])

prob = model.predict_proba(data)
print("Churn Probability:", prob[0][1])
