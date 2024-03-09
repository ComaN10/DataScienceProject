import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAnalysis:
    """
    Perform Principal Component Analysis (PCA) on a dataset using a developed method and a standard library.

    Attributes:
        X (numpy.ndarray): Input data array.
        y (numpy.ndarray): Target labels.
        num_components (int): Number of principal components.
        X_standardized (numpy.ndarray): Standardized input data.
        cov_matrix (numpy.ndarray): Covariance matrix of standardized data.
        eigenvalues (numpy.ndarray): Eigenvalues of covariance matrix.
        eigenvectors (numpy.ndarray): Eigenvectors of covariance matrix.
        pca_projection (numpy.ndarray): Projection of data onto principal components (manual PCA).
        sklearn_pca_projection (numpy.ndarray): Projection of data onto principal components (library PCA).
        sklearn_pca (sklearn.decomposition.PCA): Trained sklearn PCA object.
    """

    def __init__(self, data, targets, num_components):
        """
        Initialize PCAAnalysis with input data, targets, and number of principal components.

        Args:
            data (numpy.ndarray): Input data.
            targets (numpy.ndarray): Target labels.
            num_components (int): Number of principal components.

        Raises:
            ValueError: If the number of components exceeds the number of features.
        """
        if num_components > data.shape[1]:
            raise ValueError("Number of components cannot exceed the number of features.")

        # Dataset
        self.X = data
        self.y = targets

        # Configuration
        self.num_components = num_components

        # Plots
        self.fig, self.axes = None, None

        # Implemented PCA
        self.X_standardized = self._standardize_data()
        self.cov_matrix = self._compute_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self._compute_eigenvalues_eigenvectors()
        self.eigenvalues, self.eigenvectors = self._sort_eigenvectors()
        self.pca_projection = self._project_data()

        # Library PCA
        self.sklearn_pca_projection, self.sklearn_pca = self._apply_sklearn_pca()

    # Single underscore _: Indicates that the attribute is protected, but it's still accessible from outside the class.
    # It's a convention, not enforced by the language itself.
    # Double underscores __: Indicates that the attribute is private, and name mangling is applied. Accessing these
    # attributes from outside the class is more difficult and discouraged (example, for a method __foo the interpreter
    # replaces this name with _classname__foo).
    def _standardize_data(self):
        """
        Step 1: Standardize the dataset.
        """
        return StandardScaler().fit_transform(self.X)

    def _compute_covariance_matrix(self):
        """
        Step 2: Compute the covariance matrix.
        """
        return np.cov(self.X_standardized.T)

    def _compute_eigenvalues_eigenvectors(self):
        """
        Step 3: Compute the eigenvectors and eigenvalues.
        """
        return np.linalg.eig(self.cov_matrix)

    def _sort_eigenvectors(self):
        """
        Step 4: Sort eigenvectors based on eigenvalues.
        """
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        return self.eigenvalues[sorted_indices], self.eigenvectors[:, sorted_indices]

    def _project_data(self):
        """
        Step 5: Select the number of principal components and project the data onto them.
        """
        return self.X_standardized.dot(self.eigenvectors[:, :self.num_components])

    def _apply_sklearn_pca(self):
        """
        Apply PCA using sklearn (for comparison).
        """
        pca = PCA(n_components=self.num_components)
        return pca.fit_transform(self.X_standardized), pca

    def print_pca_projection(self):
        """
        Show the first lines of the developed and the library PCA projection.
        """
        print("PCA Projection (Manual):\n", self.pca_projection[:5])
        print("\nPCA Projection (Sklearn):\n", self.sklearn_pca_projection[:5])

    def display_feature_contributions(self):
        """
        Display feature contributions to principal components.
        """
        print("Feature Contributions to Principal Components:")
        for i, eigenvector in enumerate(self.eigenvectors.T):
            print(f"Principal Component {i + 1}:")
            for j, feature_contribution in enumerate(eigenvector):
                print(f"   Feature {j + 1}: {feature_contribution:.4f}")

    def plot_pca_projections(self):
        """
        Plot PCA projections for principal and for the two principal components in a 4 by 4 grid.
        """

        # For the developed PCA
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.axes[0, 0].scatter(self.pca_projection[:, 0], self.pca_projection[:, 0], c=self.y, cmap='viridis',
                                alpha=0.8)
        self.axes[0, 0].set_title('PCA Projection of the First Principal Component (Manual)')
        self.axes[0, 0].set_xlabel('Principal Component 1')
        self.axes[0, 0].set_ylabel('Principal Component 1')
        self.axes[0, 0].grid(True)

        self.axes[0, 1].scatter(self.pca_projection[:, 0], self.pca_projection[:, 1], c=self.y, cmap='viridis',
                                alpha=0.8)
        self.axes[0, 1].set_title('PCA Projection of the First Two Principal Components (Manual)')
        self.axes[0, 1].set_xlabel('Principal Component 1')
        self.axes[0, 1].set_ylabel('Principal Component 2')
        self.axes[0, 1].grid(True)

        # For th library PCA
        self.axes[1, 0].scatter(self.sklearn_pca_projection[:, 0], self.sklearn_pca_projection[:, 0], c=self.y,
                                cmap='viridis', alpha=0.8)
        self.axes[1, 0].set_title('PCA Projection of the First Principal Component (Sklearn)')
        self.axes[1, 0].set_xlabel('Principal Component 1')
        self.axes[1, 0].set_ylabel('Principal Component 1')
        self.axes[1, 0].grid(True)

        scatter = self.axes[1, 1].scatter(self.sklearn_pca_projection[:, 0], self.sklearn_pca_projection[:, 1],
                                          c=self.y, cmap='viridis', alpha=0.8)
        self.axes[1, 1].set_title('PCA Projection of the First Two Principal Components (Sklearn)')
        self.axes[1, 1].set_xlabel('Principal Component 1')
        self.axes[1, 1].set_ylabel('Principal Component 2')
        self.axes[1, 1].grid(True)

        # The * before scatter.legend_elements() is the unpacking operator in Python, when used before an iterable
        # (such as a list or a tuple), it unpacks the elements of the iterable into positional arguments of a function
        # or method call. In this specific context, scatter.legend_elements() returns a tuple containing two elements:
        # handles and labels. The handles represent the plotted elements (in this case, the points in the scatter plot),
        # and the labels represent the corresponding labels for those elements (in this case, the class labels). By
        # using * before scatter.legend_elements(), we are unpacking the tuple returned by scatter.legend_elements()
        # into separate arguments, which are then passed as positional arguments to the legend() method of the
        # matplotlib.axes.Axes object.
        self.axes[1, 1].add_artist(
            self.axes[1, 1].legend(*scatter.legend_elements(), title="Classes", loc="lower right"))
        plt.tight_layout()
        plt.show()

    def calculate_explained_variance_ratio(self):
        """
        Calculate explained variance ratio for both developed and library PCA.
        """
        explained_variance_ratio = self.eigenvalues[:self.num_components] / np.sum(self.eigenvalues)
        print(f"Explained Variance of the developed PCA using {self.num_components} component(s): ",
              np.sum(explained_variance_ratio))
        print(f"Explained Variance of the library PCA using {self.num_components} component(s): ",
              np.sum(self.sklearn_pca.explained_variance_ratio_))

    def plot_explained_variance_ratio(self):
        """
        Plot the explained variance ratio of the developed PCA.
        """
        explained_variance_ratio = self.eigenvalues[:self.num_components] / np.sum(self.eigenvalues)
        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(1, self.num_components + 1), explained_variance_ratio, alpha=0.5, align='center')
        for bar, value in zip(bars, explained_variance_ratio):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.2f}', ha='center',
                     va='bottom')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.title('Explained Variance Ratio per Principal Component')
        plt.grid(True)
        plt.show()