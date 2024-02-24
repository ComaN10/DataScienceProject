#import general libraries
import pandas as pd
import numpy as np
import itertools
import math

# Plotting
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed


class PlotTypes:
    """
        Contains plotting types names used in plot.
    """

    _all_types = ['hist', 'violin', 'box', 'scatter', 'lines', 'bar', 'lollypops']

    def hist(self) -> str:
        return self._all_types[0]

    def violin(self) -> str:
        return self._all_types[1]

    def box(self) -> str:
        return self._all_types[2]

    def scatter(self) -> str:
        return self._all_types[3]

    def lines(self) -> str:
        return self._all_types[4]

    def bar(self) -> str:
        return self._all_types[5]

    def lollypop(self) -> str:
        return self._all_types[6]

    def get_types(self, min: int, max: int) -> list:
        return self._all_types[min:max]

    @classmethod
    def hist_sturges(cls, data: np.array) -> float:
        return math.log(len(data),2)

    @classmethod
    def hist_freedman_diaconis(cls, data: pd.Series) -> int:
        bin_width = 2 * (data.quantile(0.75) - data.quantile(0.25)) / math.pow(len(data), 1/3)
        if bin_width == 0:
            return 10

        return (data.max() - data.min()) / bin_width

    @classmethod
    def hist_scots(cls, data: np.array) -> int:
        bin_width = 3.5 * data.std() / math.pow(len(data), 1/3)
        if bin_width == 0:
            return 10
        rest = 1 if (data.max() - data.min()) % bin_width != 0 else 0
        return ((data.max() - data.min()) // bin_width) + rest

class DataAnalysis:
    """
        Classe responsible for showing up the data, making it easier to analyze data sparsity, distribution, ... TODO: add docstring
    """

    def __init__(self,file="fetal_health.csv"):
        """
            Initialize the object with the data from the file
            :param file: name of the csv file to load and analise. Default is fetal_health.csv
        """
        self.file = file
        self.dataset = pd.read_csv(self.file)

    def save_as_csv(self) -> None:
        """
            Save the data as a csv file
            :return:
        """
        self.dataset.to_csv(self.file)

    def show_datainfo(self) -> None:
        """
            Shows info about the loaded data. Basic info, description, data dispersion graphs.
            :return: None
        """
        print("Info")
        self.info()
        print()
        print("Description")
        self.describe()
        print()

    def pre_process(self):
        """
            Removes duplicated, Remove null values, normalize, standardization
            :return None:
        """
        self.remove_duplicates()
        self.remove_null_values()
        self.normalize()
        self.standardization()

    def remove_duplicates(self) -> None:
        """
            Removes duplicated values and missing values
            :return None:
        """
        print("Count before removing duplicates ", len(self.dataset))
        self.dataset.drop_duplicates(inplace=True)
        print("Count after removing duplicates ", len(self.dataset))

    def remove_null_values(self) -> None:
        """
            Removes duplicated values and missing values
            :return None:
        """
        print("Count before ", len(self.dataset))
        self.dataset.dropna(inplace=True)
        print("Count after drop rows with null values ", len(self.dataset))

    def normalize(self) -> None:
        """
            Applies min/max normalization to each column of the dataset
            and replaces the actual feature data with the normalized data.
            The result data is between 0 and 1.
            :return None:
        """

        for column in self.dataset.columns:
            min_value = self.dataset[column].min()
            max_value = self.dataset[column].max()
            self.dataset[column] = (self.dataset[column] - min_value) / (max_value - min_value)

    def standardization(self) -> None:
        """
            Applies standardization to each column of the dataset
            and replaces the actual feature data with the standardized data.
            :return None:
        """

        for column in self.dataset.columns:
            mean = self.dataset[column].mean()
            std = self.dataset[column].std()
            self.dataset[column] = (self.dataset[column] - mean) / std


    def describe(self) -> None:
        """
            Show the description of the loaded dataset: Feature mean median std min max q25 q50 q75
            :return: None
        """
        print(self.dataset.describe())

        # for colum in self.dataset.columns:
        #
        #     print("Feature ",colum)
        #     values = self.dataset[colum]
        #     print("mean ", values.mean())
        #     print("median ", values.median())
        #     print("std ", values.std())
        #     print("min ", values.min())
        #     print("max ", values.max())
        #     print("Q25 ", values.quantile(0.25))
        #     print("Q50 ", values.quantile(0.50))
        #     print("Q75 ", values.quantile(0.75), end="\n\n")

    def info(self) -> None:
        """
            Print information about the loaded dataset
            :return None:
        """
        self.dataset.info()

    def plot_features(self, plot_types: list, columns=4, hist_number_of_bars_func=lambda x: 10, plot_size=5) -> None:
        """
            Plot features per class label using matplotlib, and it plots the plot_types specified in plot_types.

            :param plot_types: An array specifying the type of plot to generate for each feature.Supported plot types: All in PlotTypes
            :param columns: Number of features to per row
            :param hist_number_of_bars_func: function that return the number of bins to the histogram based on the data
            :param plot_size: Size of each plot

            :return None:
        """

        target = "fetal_health"

        # For each plot, need to produce a subplot for each feature, and then, for the current subplot, check the data
        # related to each label
        for plot_type in plot_types:
            # Create a figure with a single row and subplots for each feature in different columns, with width of 15
            # inches and height of 5 inches, fig represents the entire figure, while axes is an array of axes objects
            # representing each subplot (in this case it will be a one-dimensional array containing references to each
            # subplot)
            number_of_features = len(self.dataset.columns)
            rows = number_of_features//columns
            number_of_rows = rows if number_of_features % columns == 0 else rows + 1

            fig, axes = plt.subplots(nrows=number_of_rows, ncols=columns, figsize=(rows*plot_size, columns*plot_size))

            for i, feature in enumerate(self.dataset.columns):

                # Produce the subplot for each feature
                # two-dimensional array containing references to each subplot
                ax = axes[i // columns][i % columns]
                hist_bins = int(hist_number_of_bars_func(self.dataset[feature]))
                ax.set_title(feature)

                # for each classification in target draw a graph in the same feature
                for label in self.dataset[target].unique():

                    # select the data of the class label and then get the feature data
                    feature_data = self.dataset[self.dataset[target] == label][feature]

                    # Plot according to the specified plot type
                    if plot_type == 'hist':
                        ax.hist(feature_data, bins=hist_bins, alpha=0.25)
                        # Set plot labels
                        ax.set_xlabel(f'{feature}')
                        ax.set_ylabel(f'Frequency')

                    elif plot_type == 'violin' and target != "":
                        ax.violinplot(feature_data, showmeans=True, showmedians=True, positions=[label])
                        # Set plot title and labels
                        ax.set_xlabel("Class")
                        ax.set_ylabel("Values")

                    elif plot_type == 'box':
                        ax.boxplot(feature_data, vert=True, positions=[label])
                        ax.set_xlabel("Class")
                        ax.set_ylabel("Values")

                    elif plot_type == 'scatter':
                        # Define the x axis line to go from 0 to len(feature_data) (because first label is 0) then from
                        # then from len(feature_data) to 2*len(feature_data) (because first label is 1) and so one
                        x_axis = np.arange(len(feature_data)) + len(feature_data) * label
                        ax.scatter(x_axis, feature_data, label=f'Class {int(label)}')
                        ax.legend()

                    elif plot_type == 'lines':
                        ax.plot(feature_data, label=f'Class {int(label)}')
                        ax.legend()

                    elif plot_type == 'bar':
                        x_axis = np.arange(len(feature_data)) + len(feature_data) * label
                        ax.bar(x_axis, feature_data, label=label)
                        ax.legend()

                    elif plot_type == 'lollypops':
                        x_axis = np.arange(len(feature_data)) + len(feature_data) * label
                        ax.stem(x_axis, feature_data, linefmt='-', label=label)
                        ax.legend()

            # Adjust layout
            plt.tight_layout()
            # Show plot
            plt.show()
