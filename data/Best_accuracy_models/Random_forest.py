from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import  accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,roc_auc_score,roc_curve,auc

data=pd.read_csv("data/processed/final_dataset.csv")

X = data.drop(columns=['Customer ID', 'Churn'])
y = data['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

random_forest_model=RandomForestClassifier(n_estimators=100)

random_forest_model.fit(X_train,y_train)

random_forest_model_pred=random_forest_model.predict(X_test)

print("Predicted:", random_forest_model_pred)
print("Actual:", y_test.values)

print("Random Forest Accuracy:", accuracy_score(y_test, random_forest_model_pred))

cm = confusion_matrix(y_test, random_forest_model_pred)
print("\nConfusion Matrix:\n", cm)

cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cmd.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

print("\nClassification Report:\n", classification_report(y_test, random_forest_model_pred))

#AUC-ROC
y_probs=random_forest_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print("\nROC-AUC Score:", roc_auc)
# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--') 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve -Random Forest")
plt.legend()
plt.show()



