import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,roc_auc_score,auc,roc_curve

data=pd.read_csv("data/processed/final_dataset.csv")

X = data.drop(columns=['Customer ID', 'Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Base model (Decision Stump)-simple dec tree
dt_model = DecisionTreeClassifier(max_depth=1)

# AdaBoost Model
adaboost_model = AdaBoostClassifier(
    estimator=dt_model,
    n_estimators=100,
    random_state=42
)

adaboost_model.fit(X_train, y_train)

adaboost_pred = adaboost_model.predict(X_test)

print("Predicted:", adaboost_pred)
print("Actual:", y_test.values)

print("AdaBoost Accuracy:", accuracy_score(y_test, adaboost_pred))

cm = confusion_matrix(y_test, adaboost_pred)
print("\nConfusion Matrix:\n", cm)

cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cmd.plot()
plt.title("AdaBoost Confusion Matrix")
plt.show()

print("\nClassification Report:\n", classification_report(y_test, adaboost_pred))

# AUC-ROC
y_probs = adaboost_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

print("\nROC-AUC Score:", roc_auc)

# ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - AdaBoost")
plt.legend()
plt.show()

