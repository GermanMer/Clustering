import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar datos aleatorios con make_blobs
n_samples = 300
n_features = 2
n_clusters = 3
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Crear un objeto KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init=10 indica que el algoritmo se ejecutará 10 veces con diferentes inicializaciones aleatorias de centroides y seleccionará la mejor inicialización basada en la inercia.

# Ajustar el modelo K-Means a los datos
kmeans.fit(X)

# Obtener las etiquetas de cluster asignadas a cada punto de datos
labels = kmeans.labels_

# Obtener las coordenadas de los centroides
centroids = kmeans.cluster_centers_

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r', label='Centroides')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Ejemplo de K-Means Clustering')
plt.legend()
plt.show()
