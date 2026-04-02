import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,roc_auc_score,roc_curve,auc
from sklearn.tree import plot_tree

#Decision Tree->scaling not reqd
data = pd.read_csv("data/processed/final_dataset.csv")

# Define X and y
X = data.drop(columns=['Customer ID', 'Churn'])
y = data['Churn']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
    criterion='gini',     # split quality
    max_depth=5,          # controls overfitting
    random_state=42
)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("Predicted:", dt_pred)
print("Actual:", y_test.values)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

cm = confusion_matrix(y_test, dt_pred)
print("\nConfusion Matrix:\n", cm)

cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cmd.plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

print("\nClassification Report:\n", classification_report(y_test, dt_pred))

plt.figure(figsize=(15, 10))

plot_tree(
    dt_model,
    feature_names=X.columns,          
    class_names=["No Churn", "Churn"], 
    filled=True,                      # color nodes
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree for Customer Intelligence & Churn Prediction")
plt.show()

#AUC-ROC
y_probs=dt_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print("\nROC-AUC Score:", roc_auc)
# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--') 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve -Decision Tree")
plt.legend()
plt.show()