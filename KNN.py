import numpy as np

class KNN:
    """
    K-Nearest Neighbors (kNN) classifier implemented using numpy arrays.

    Parameters:
    - k: int, optional (default=5)
        Number of nearest neighbors to consider during classification.
    """

    def __init__(self, k=5):
        """
        Initialize the KNN classifier.

        Parameters:
        - k: int, optional (default=5)
            Number of nearest neighbors to consider during classification.
        """
        self.k = k

    def fit(self, X_train, y_train):
        """
        Train the KNN classifier.

        Parameters:
        - X_train: numpy array, shape (n_samples, n_features)
            Training data.
        - y_train: numpy array, shape (n_samples,)
            Labels corresponding to the training data.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, x_test):
        """
        Predict the class labels for test data.

        Parameters:
        - X_test: numpy array, shape (n_samples, n_features)
            Test data.

        Returns:
        - y_pred: numpy array, shape (n_samples,)
            Predicted class labels for the test data.
        """
        y_pred = np.empty(x_test.shape[0], dtype=self.y_train.dtype)

        # Iterate over each test sample
        for i, sample in enumerate(x_test):

            # Calculate Euclidean distances between the test sample and all training samples
            distances = np.linalg.norm(self.X_train - sample, axis=1)

            # Find the indices of the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]

            # Get the labels of the k nearest neighbors
            nearest_labels = self.y_train[nearest_indices]

            # Predict the class label based on majority vote
            y_pred[i] = np.bincount(nearest_labels).argmax()

        return y_pred