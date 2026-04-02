import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

data=pd.read_csv("data/processed/final_dataset.csv")

X=data.drop(columns=['Customer ID','Churn'])

#find optimal clusters
linkage_data=linkage(X, method='ward')

plt.figure(figsize=(8,5))
dendrogram(linkage_data)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

Hierarchical_clu = AgglomerativeClustering(n_clusters=3, linkage='ward')

data['HC_Cluster'] = Hierarchical_clu.fit_predict(X)

plt.scatter(data['Tenure'], data['MonthlyCharges'], c=data['HC_Cluster'])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Customer Segmentation (Hierarchical Clustering)")
plt.show()
print(data.groupby('HC_Cluster')['Churn'].mean())

score_hc = silhouette_score(X, data['HC_Cluster'])
print("Hierarchical Silhouette Score:", score_hc)
