import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Customer_Info = pd.read_csv("data/raw/Customer_Info.csv")
Working=pd.read_csv("data/raw/Working.csv")
Services = pd.read_csv("data/raw/Services.csv")
Payment= pd.read_csv("data/raw/Payment.csv")
Contract_Churn = pd.read_csv("data/raw/Contract_Churn.csv")

# Merge the datasets 
data=Customer_Info.merge(Working, on='Customer ID')
data=data.merge(Services, on='Customer ID')
data=data.merge(Payment, on='Customer ID')
data=data.merge(Contract_Churn, on='Customer ID')

data.to_csv("data/processed/merged_data.csv", index=False)
print("Data Merged and Saved Successfully!")

#data understanding
print(data.shape)
print(data.columns)
print(data.info())
print(data.head())

#data cleaning
#handle missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

#summarizes the data
print("\nStatistical Summary (Numerical):\n")
print(data.describe())
print("\nStatistical Summary (Categorical):\n")
data.describe(include=['object', 'string'])

#encoding -by label or  by mapping 
from sklearn.preprocessing import LabelEncoder

for col in ["Gender","Partner","Dependents","PhoneService","MultipleLines",
            "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
            "StreamingTV","StreamingMovies","PaperlessBilling","Churn"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

from sklearn.preprocessing import OneHotEncoder
data=pd.get_dummies(data, columns=["InternetService","Contract", "PaymentMethod"], drop_first=True)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(data.columns)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')#coerce-replace invalid-nan
data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce')
data=data.dropna()

cols_to_scale = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
print("Scaling Applied Successfully!")

data.to_csv("data/processed/final_dataset.csv", index=False)
print("Data Preprocessed and Saved Successfully!")
