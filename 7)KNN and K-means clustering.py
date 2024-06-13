import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for _ in range(self.max_iters):
            # Assign each sample to the nearest centroid
            clusters = [[] for _ in range(self.k)]
            for x in X:
                closest_centroid_idx = np.argmin([np.linalg.norm(x - centroid) for centroid in self.centroids])
                clusters[closest_centroid_idx].append(x)

            # Update centroids
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in clusters]

            # Check for convergence
            if np.allclose(prev_centroids, self.centroids):
                break

    def predict(self, X):
        return np.array([np.argmin([np.linalg.norm(x - centroid) for centroid in self.centroids]) for x in X])
X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train = np.array([0, 0, 1, 1, 0, 1])
X_test = np.array([[1, 3], [8, 9], [0, 3], [5, 4]])
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(predictions) 
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
kmeans = KMeans(k=2)
kmeans.fit(X)
clusters = kmeans.predict(X)
print(clusters) 
