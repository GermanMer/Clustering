import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data

# Calcular la matriz de enlace utilizando el método de enlace completo
linkage_matrix = linkage(X, method='complete')

# Generar el dendrograma
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, orientation='top', labels=iris.target, leaf_rotation=90)
plt.title('Dendrograma de Hierarchical Clustering')
plt.xlabel('Muestras de Iris')
plt.ylabel('Distancia')
plt.show()

# Realizar corte en el dendrograma para obtener clusters
k = 3  # Número de clusters deseados
cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')

# Calcular la matriz de distancias originales
original_dist = pdist(X)

# Calcular el Coeficiente de Correlación Cofenética
coph_corr, _ = cophenet(linkage_matrix, original_dist)
print(f'Coeficiente de Correlación Cofenética: {coph_corr}')

# Calcular el Índice de Silueta
silhouette_avg = silhouette_score(X, cluster_labels)
print(f'Índice de Silueta: {silhouette_avg}')

# Mostrar las etiquetas de cluster resultantes
print("Etiquetas de Cluster:")
print(cluster_labels)

# Visualizar los clusters resultantes
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Clusters después del Corte')
plt.show()

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
