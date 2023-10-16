import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Generar datos de ejemplo con dos lunas
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Definir una gama de valores para eps y min_samples
eps_values = np.linspace(0.1, 0.5, 5)
min_samples_values = range(2, 11)

best_ch_score = -1  # Inicializar con un valor bajo
best_eps = None
best_min_samples = None

for eps in eps_values:
    for min_samples in min_samples_values:
        # Crear un objeto DBSCAN con los parámetros actuales
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        # Ajustar el modelo a los datos
        dbscan.fit(X)

        # Obtener las etiquetas de cluster asignadas a cada punto de datos
        labels = dbscan.labels_

        # Ignorar los resultados que generan un solo cluster o ruido
        if len(set(labels)) <= 2:
            continue

        # Calcular la métrica de Silueta
        silhouette_avg = silhouette_score(X, labels)

        # Calcular la métrica de Calinski-Harabasz
        ch_score = calinski_harabasz_score(X, labels)

        # Actualizar los mejores parámetros si encontramos un mejor resultado
        if ch_score > best_ch_score:
            best_ch_score = ch_score
            best_eps = eps
            best_min_samples = min_samples
            best_labels = labels  # Actualizar las etiquetas del mejor modelo

print(f'Mejores parámetros - eps: {best_eps}, min_samples: {best_min_samples}')

# Visualizar los resultados
plt.figure(figsize=(8, 6))
unique_labels = set(best_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Puntos de ruido en negro

    class_member_mask = (best_labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8, label=f'Cluster {k}')

# Calcular la métrica de Silueta y el índice de Calinski-Harabasz del mejor modelo
best_silhouette_avg = silhouette_score(X, best_labels)
best_ch_score = calinski_harabasz_score(X, best_labels)

plt.title(f'Número estimado de clusters: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}\nSilueta: {best_silhouette_avg:.2f}\nCalinski-Harabasz: {best_ch_score:.2f}')
plt.legend(loc='upper right')
plt.show()
