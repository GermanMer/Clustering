import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Generar datos de ejemplo con dos lunas
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Crear un objeto DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)

# Ajustar el modelo a los datos
dbscan.fit(X)

# Obtener las etiquetas de cluster asignadas a cada punto de datos
labels = dbscan.labels_

# Número de clusters encontrados (-1 representa puntos de ruido)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

# Calcular la métrica de Silueta
silhouette_avg = silhouette_score(X, labels)

# Calcular la métrica de Calinski-Harabasz
ch_score = calinski_harabasz_score(X, labels)

# Visualizar los resultados
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Puntos de ruido en negro

    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8, label=f'Cluster {k}')

plt.title(f'Número estimado de clusters: {n_clusters}\nSilueta: {silhouette_avg:.2f}\nCalinski-Harabasz: {ch_score:.2f}')
plt.legend(loc='upper right')
plt.show()
