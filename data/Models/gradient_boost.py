import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,roc_auc_score,auc,roc_curve

data=pd.read_csv("data/processed/final_dataset.csv")

X = data.drop(columns=['Customer ID', 'Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Gradient Boosting Model
gradient_boost_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    max_depth=3  #manually add tree
)

gradient_boost_model.fit(X_train, y_train)

gradient_pred = gradient_boost_model.predict(X_test)

print("Predicted:", gradient_pred)
print("Actual:", y_test.values)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, gradient_pred))

cm = confusion_matrix(y_test, gradient_pred)
print("\nConfusion Matrix:\n", cm)

cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
cmd.plot()
plt.title("Gradient Boosting Confusion Matrix")
plt.show()

print("\nClassification Report:\n", classification_report(y_test, gradient_pred))

# AUC-ROC
y_probs = gradient_boost_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

print("\nROC-AUC Score:", roc_auc)

# ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Gradient Boosting")
plt.legend()
plt.show()