import numpy as np
import matplotlib.pyplot as plt
# for high-dimensional data reduction, noise reduction, and feature selection
from sklearn.decomposition import PCA
# for EDA and visualization, revealing non-linear structures in high-dimensional data
from sklearn.manifold import TSNE
# supervised learning tasks to maximize class separability and for dimensionality reduction with class information
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# for emphasizing preservation of local relationships and non-linear structures
from sklearn.manifold import LocallyLinearEmbedding
# for visualization balancing preservation of both local and global structures

from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
import itertools
import umap


class DimensionalityReduction:
    """
        Classe for making dimensionality reduction
    """
    def __init__(self, data, targets, standardized=False):
        """
            Initialize the DimensionalityReduction object with the dataset.

            :param data: The dataset to perform dimensionality reduction on.
            :param targets: The targets of the samples.
            :param standardized: If true means that passed data is already standardize.
        """
        self.data = StandardScaler().fit_transform(data) if not standardized else data
        self.targets = targets

    def compute_pca(self, n_components=2):
        """
            Compute Principal Component Analysis (PCA) on the dataset.

            :param n_components: The number of components to keep.
            :return pca_projection: The projected data using PCA.
        """
        return PCA(n_components=n_components).fit_transform(self.data)

    def compute_lda(self, n_components=2):
        """
            Perform Linear Discriminant Analysis (LDA) on the input data.

            :param n_components: The number of components to keep.
            :return array-like: The reduced-dimensional representation of the data using LDA.
        """
        return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(self.data, self.targets)

    def compute_tsne(self, n_components=2, perplexity=3):
        """
            Compute t-Distributed Stochastic Neighbor Embedding (t-SNE) on the dataset.

            :param n_components: The number of components to embed the data into.
            :param perplexity: The perplexity parameter for t-SNE.

            :return tsne_projection: The projected data using t-SNE.
        """
        return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(self.data)

    def compute_umap(self, n_components=2, n_neighbors=8, min_dist=0.5, metric='euclidean'):
        """
            Compute Uniform Manifold Approximation and Projection (UMAP) on the dataset.

            :param n_components: The number of components to embed the data into.
            :param n_neighbors: The number of neighbors to consider for each point.
            :param min_dist: The minimum distance between embedded points.
            :param metric: The distance metric to use.

            :return umap_projection: The projected data using UMAP.
        """
        #Todo:Resolve umap import
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric).fit_transform(self.data)

    def compute_lle(self, n_components=2, n_neighbors=20):
        """
            Compute Locally Linear Embedding (LLE) on the dataset.

            :param n_components: The number of components to embed the data into.
            :param n_neighbors: The number of neighbors to consider for each point.

            :return lle_projection: The projected data using LLE.
        """
        return LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components).fit_transform(self.data)

    def plot_projection(self, projection, title):
        """
            Plot the 2D projection of the dataset.

            :param projection: The projected data.
            :param title: The title of the plot.

            :return fig: The
        """

        plt.figure(figsize=(8, 6))
        plt.scatter(projection[:, 0], projection[:, 1], c=self.targets, alpha=0.25)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

    def plot_3d_combinations(self, begin,end):
        """
            Plot 3D scatter plots for each combination of three features using Plotly.
        """

        if len(self.data.columns) < 3:
            raise ValueError("This method requires at least three features to plot in 3D.")

        if begin < 0 or end < 0:
            raise ValueError("This method requires begin >= 0 , end > begin")

        if end <= begin:
            raise ValueError("his method requires end > begin")

        # Produce all possible unique combinations of three features
        feature_combinations = itertools.combinations(self.data.columns, 3)
        itera = 0
        # Do a plot for each possible combination of three features
        for combination in feature_combinations:

            if itera == end:
                break
            elif itera < begin:
                continue

            print("Ploting ", itera, )

            itera += 1

            trace = go.Scatter3d(
                x=self.data[combination[0]],
                y=self.data[combination[1]],
                z=self.data[combination[2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.targets,
                    colorscale='Viridis',
                    opacity=0.8
                )
            )

            # Define the layout of the plot presentation
            layout = go.Layout(
                scene=dict(
                    xaxis=dict(title=combination[0]),
                    yaxis=dict(title=combination[1]),
                    zaxis=dict(title=combination[2])
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )

            fig = go.Figure(data=[trace], layout=layout)
            fig.show()
