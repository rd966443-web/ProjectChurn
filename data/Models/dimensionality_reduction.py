import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data/processed/final_dataset.csv")

X=data.drop(columns=['Customer ID','Churn'])
y=data['Churn']

#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # keep 95% variance
X_pca = pca.fit_transform(X)
print("\nNumber of PCA components:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Scatter Plot")
plt.colorbar(label="Churn")
plt.show()

#Chi-square
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Chi-Square requires non-negative features So-MinMaxScaler 0-1
from sklearn.preprocessing import MinMaxScaler
X_minmax = MinMaxScaler().fit_transform(X) 
chi_scores, p_values = chi2(X_minmax, y)

#for feature importance
chi_df = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi_scores,
    'p-value': p_values
}).sort_values(by='Chi2 Score', ascending=False)

significant_features = chi_df[chi_df['p-value'] < 0.05]['Feature']#for interpret
print("Significant Features:", list(significant_features))

top_k = 20#for predictive
selector = SelectKBest(score_func=chi2, k=top_k)
X_selected = selector.fit_transform(X_minmax, y)
selected_features = X.columns[selector.get_support()]#names
print("Top Features:",selected_features)

#top_k is best for correlated
#significant level-sometimes leaves imp features 

#hybrid approach-combo for both

# Keep features that are in Top K AND significant
significant_topk = chi_df[(chi_df['Feature'].isin(selected_features)) & (chi_df['p-value'] < 0.05)]
final_features = list(significant_topk['Feature'])
print("Hybrid Selected Features:", final_features)





