import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generar datos aleatorios con make_blobs
n_samples = 300
n_features = 2
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=4, random_state=42)

# Rango de valores de K que deseas probar
k_values = range(2, 10)

# Listas para almacenar las métricas de evaluación
inertia_values = []
silhouette_values = []

for k in k_values:
    # Crear un objeto KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Ajustar el modelo K-Means a los datos
    kmeans.fit(X)

    # Calcular la inercia y el coeficiente de silueta
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(X, kmeans.labels_)

    # Agregar los valores a las listas
    inertia_values.append(inertia)
    silhouette_values.append(silhouette_avg)

# Graficar la inercia en función de K
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.title('Método del Codo')

# Graficar el coeficiente de silueta en función de K
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_values, marker='o', linestyle='-', color='g')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Coeficiente de Silueta')
plt.title('Método de la Silueta')

plt.tight_layout()
plt.show()
