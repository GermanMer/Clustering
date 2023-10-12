import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data

# Calcular la matriz de enlace utilizando el método de enlace completo
linkage_matrix = linkage(X, method='complete')

# Generar el dendrograma
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, orientation='top', labels=iris.target, leaf_rotation=90)
plt.title('Dendrograma de Hierarchical Clustering (Divisivo)')
plt.xlabel('Muestras de Iris')
plt.ylabel('Distancia')
plt.show()

# Realizar corte en el dendrograma para obtener clusters
k = 3  # Número de clusters deseados
cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')

# Mostrar las etiquetas de cluster resultantes
print("Etiquetas de Cluster:")
print(cluster_labels)
