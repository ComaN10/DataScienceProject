import pandas as pd
from DataAnalysis import DataAnalysis


class FeatureEngineering:
    """
        TODO: add docstring
    """

    def __init__(self,data: DataAnalysis):
        self.dataAnalysis = data
        self.dataset = self.dataAnalysis.dataset.iloc[:, :-1]
    def create_default_features(self) -> None:
        """
            Creates the default features. Mean,Std,min,max,median ...
            :return None:
        """
        # Calculate the mean by row
        self.dataAnalysis.dataset['row_mean'] = self.dataAnalysis.dataset.mean(axis=1)

        # Calculate the median by row
        self.dataAnalysis.dataset['row_median'] = self.dataAnalysis.dataset.median(axis=1)

        # Calculate the min by row
        self.dataAnalysis.dataset['row_min'] = self.dataAnalysis.dataset.min(axis=1)

        # Calculate the max by row
        self.dataAnalysis.dataset['row_max'] = self.dataAnalysis.dataset.max(axis=1)

        # Calculate the variance by row
        self.dataAnalysis.dataset['row_variance'] = self.dataAnalysis.dataset.var(axis=1)

    def create_all_features(self) -> None:
        """
            Creates features based on the base features of de dataset
            :return:none
        """

        self.create_default_features()

        # Interaction between "Uterine Contractions" and "Fetal Movement,"
        dataset = self.dataAnalysis.dataset.iloc[:, :-1]
        dataset['interaction_uterine_fetal'] = dataset['uterine_contractions'] * dataset['fetal_movement']

        # Interaction between "Fetal Movement" and "accelerations"
        dataset['interaction_movment_accelerations']= dataset['fetal_movement'] * dataset['accelerations']

        # Calculate and store the rolling mean of 'accelerations' with a specified time window.
        time_window = 3
        dataset['rolling_mean_accelerations'] = dataset['accelerations'].rolling(window=time_window).mean()

        # Calculate the Skewness by row
        # If skewness is greater than 0, indicates that the right tail of the distribution
        # is longer or fatter than the left tail.
        # If it lower than 0 then the right tail is longer or flatter and if 0 is normal distribution

        dataset['row_skewness'] = dataset.skew(axis=1)
        # Calculate the Kurtosis by row
        dataset['row_kurtosis'] = dataset.kurtosis(axis=1)
        # Define the bins and labels for the categorical binning
        bins = [0, 50, 100, 150, float('inf')]  # Adjust the bin edges as needed
        labels = ['Low', 'Medium', 'High', 'Very High']
        # Perform categorical binning for a specific column, e.g., 'accelerations'
        dataset['accelerations_category'] = pd.cut(dataset['accelerations'], bins=bins, labels=labels, include_lowest=True)
        # Display the DataFrame with the new 'bmi' feature
        df_except_last = dataset.iloc[:, :-1]
        dataset['sum_rows']=df_except_last.sum(axis=1)
        print(dataset['sum_rows'].describe())
        print(dataset)






