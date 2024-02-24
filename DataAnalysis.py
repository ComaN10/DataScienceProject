import pandas as pd
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed


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

    def save_as_csv(self):
        """
        Save the data as a csv file
        :return:
        """
        self.dataset.to_csv(self.file)

    def show_datainfo(self):
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

    def describe(self):
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

    def info(self):
        """
        Print information about the loaded dataset
        :return: None
        """
        self.dataset.info()
