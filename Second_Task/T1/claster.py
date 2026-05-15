from itertools import combinations
import numpy as np

def find_best_cluster(Psi, cluster_size=5):
    """Находит cluster_size наиболее близких точек"""
    n = Psi.shape[0]
    X = Psi[:, 1:]  # только предикторы, без константы
    
    best_cluster = None
    min_avg_distance = float('inf')
    
    # Перебираем все комбинации из 5 точек
    for combo in combinations(range(n), cluster_size):
        # Вычисляем среднее попарное расстояние
        distances = []
        for i, j in combinations(combo, 2):
            dist = np.linalg.norm(X[i] - X[j])
            distances.append(dist)
        
        avg_dist = np.mean(distances)
        
        if avg_dist < min_avg_distance:
            min_avg_distance = avg_dist
            best_cluster = combo
    
    return best_cluster, min_avg_distance

# Поиск лучшего кластера
best_idx, avg_dist = find_best_cluster(Psi, cluster_size=5)
print(f"Лучший кластер: индексы {best_idx}")
print(f"Среднее попарное расстояние: {avg_dist:.4f}")
print("\nКоординаты точек:")
print(Psi[list(best_idx)])