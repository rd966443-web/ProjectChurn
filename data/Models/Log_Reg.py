import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,roc_auc_score,roc_curve,auc
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

data = pd.read_csv("data/processed/final_dataset.csv")

# Define X and y
X = data.drop(columns=['Customer ID', 'Churn'])
y = data['Churn']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

coefficients=model.coef_[0]
intercept=model.intercept_[0]
print("Coefficients:",coefficients)
print("Intercept:",intercept)

# Predictions
y_pred= model.predict(X_test)
print("Predicted:",y_pred)
print("Actual:",y_test.values)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n",cm)
cmd=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
cmd.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# PCA (Principal Component Analysis)->used becoz of many features
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# get pred again->clr var
model_pred = model.predict(X_test)

plt.figure(figsize=(10,6))
plt.scatter(
    X_test_pca[model_pred == 0, 0],
    X_test_pca[model_pred == 0, 1],
    label="Predicted No Churn",
    alpha=0.6
)
plt.scatter(
    X_test_pca[model_pred == 1, 0],
    X_test_pca[model_pred == 1, 1],
    label="Predicted Churn",
    alpha=0.6
)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Logistic Regression Predictions (PCA View)")
plt.legend()
plt.show()

# AUC-ROC
y_probs=model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print("\nROC-AUC Score:", roc_auc)
# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--') 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve -Logistic Regression")
plt.legend()
plt.show()

#RFE--recursive feature elimination
rfe = RFE(estimator=model, n_features_to_select=20)
rfe.fit(X_train, y_train)

y_pred = rfe.predict(X_test)
print("Selected features:", rfe.support_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))