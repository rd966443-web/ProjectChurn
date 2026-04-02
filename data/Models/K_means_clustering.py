import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("data/processed/final_dataset.csv")

X=data.drop(columns=['Customer ID','Churn'])

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

#elbow-3-->2-3  between not good
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

plt.scatter(data['Tenure'], data['MonthlyCharges'], c=data['Cluster'])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Customer Segmentation (K-Means)")
plt.show()

#centroid
print("Cluster Centers:\n", kmeans.cluster_centers_)
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker='X',
    s=200
)
plt.show()

print(data.groupby('Cluster')['Churn'].mean())

from sklearn.metrics import silhouette_score

score = silhouette_score(X, data['Cluster'])
print("Silhouette Score:", score)

#clustered data-k-means not good 