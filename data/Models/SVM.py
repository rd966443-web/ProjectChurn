import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,roc_auc_score,roc_curve,auc,classification_report

data=pd.read_csv("data/processed/final_dataset.csv")

X = data.drop(columns=['Customer ID', 'Churn'])
y = data['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# SVM Model
svm_model = svm.SVC(kernel='rbf', probability=True, random_state=42)#prob-roc_auc#rbf-separation

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

print("\nPredicted:", svm_pred)
print("Actual:", y_test.values)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

cm = confusion_matrix(y_test, svm_pred)
print("\nConfusion Matrix:\n", cm)

cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cmd.plot()
plt.title("SVM Confusion Matrix")
plt.show()

print("\nClassification Report:\n", classification_report(y_test, svm_pred))

# AUC-ROC
y_probs = svm_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

print("\nROC-AUC Score:", roc_auc)

# ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.show()