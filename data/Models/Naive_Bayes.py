import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,auc

data=pd.read_csv("data/processed/final_dataset.csv")

# Define X and y
X = data.drop(columns=['Customer ID', 'Churn'])
y = data['Churn']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X.dtypes)

naive_bayes=GaussianNB()
naive_bayes.fit(X_train,y_train)

y_pred= naive_bayes.predict(X_test)                                                             
print("Predicted:",y_pred)
print("Actual:",y_test.values)

# Evaluation
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n",cm)
cmd=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
cmd.plot()
plt.title("Naive Bayes Confusion Matrix")
plt.show()
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#AUC-ROC
y_probs= naive_bayes.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print("\nROC-AUC Score:", roc_auc)
# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--') 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Naive Bayes")
plt.legend()
plt.show()