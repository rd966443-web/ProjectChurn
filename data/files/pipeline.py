import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report,accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Data load
data=pd.read_csv("data/processed/final_data.csv")

#data cleaning
#Remove infinite values
data.replace([np.inf,-np.inf],np.nan,inplace=True)
#drop nan 
data.dropna(inplace=True)

X = data.drop(columns=['Customer ID','Churn'])
y = data['Churn'].map({"Yes":1, "No":0})

#save col for streamlit
joblib.dump(X.columns,"columns.pkl")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#preprocessing 
num_cols = X.select_dtypes(include=['int64', 'float64']).columns 
cat_cols = X.select_dtypes(include=['object', 'bool']).columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([ 
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler()) 
        ]), num_cols), 
    ("cat", Pipeline([ 
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("encoder", OneHotEncoder(handle_unknown="ignore")) 
        ]),cat_cols) 
])

#models used

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42, learning_rate=0.5)
}
# SAMME.R uses probabilities
# SAMME uses class predictions
# sklearn is simplifying things going forward

# Define hyperparameter grids for models
# param_grids is a dict jis mein hyperparameters ki value hoti hai for model
param_grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10]
    },
    "Random Forest": {
        "model__max_depth": [3, 5, 7]
    },
    "AdaBoost": {
        "model__learning_rate": [0.01, 0.1, 0.5]
    }
}
#c-less,C-more for regularization    

#Training
best_score = 0
best_model_final = None

for name, model in models.items():
    print(f"\n Training Model:{name}")

#pipeline with preprocessing
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    #Use pipeline (not Pipeline class)
    grid = GridSearchCV(pipeline, param_grids[name], cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    #best estimator
    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    probs = best_model.predict_proba(X_test)[:,1]

    acc=accuracy_score(y_test,preds)
    roc = roc_auc_score(y_test, probs)
    
    print("Accuracy Score",acc)
    print("ROC AUC:", roc)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("Classification Report:")
    print(classification_report(y_test, preds))

    if roc > best_score:
        best_score = roc
        best_model_final = best_model

# Feature Importance
# Safely handle both pipeline and raw model
if isinstance(best_model_final, Pipeline):
    model_obj = best_model_final.named_steps["model"]
    try:
        feature_names = best_model_final.named_steps["preprocessing"].get_feature_names_out()
    except:
        feature_names = X.columns
else:
    model_obj = best_model_final
    feature_names = X.columns

# Feature importance only for tree-based models
if hasattr(model_obj, "feature_importances_"):
    importances = model_obj.feature_importances_
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print("\nTop Features:")
    print(feature_importance.head(20))
else:
    print("Feature importance not available for this model.")

# Save by own after check
# if name == "AdaBoost":
#     joblib.dump(best_model, "adaboost_model.pkl")

        #automatically saved

# Save Final Model 
joblib.dump(best_model_final, "bestt_model.pkl")
print("Best model saved successfully!")


