from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K)
kmeans.fit(X)
inertia = kmeans.inertia_
