import numpy as np
import pandas as pd
from DataAnalysis import DataAnalysis


class FeatureEngineering:
    """
        TODO: add docstring
    """

    categories = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4, "Very High": 5}

    @classmethod
    def _create_default_features(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        """
            Creates the default features. Mean,Std,min,max,median ...
            :param dataset: The dataset to be used for adding features
            :return None:
        """

        # Calculate the mean by row
        dataset['row_mean'] = dataset.mean(axis=1)

        # Calculate the median by row
        dataset['row_median'] = dataset.median(axis=1)

        # Calculate the min by row
        dataset['row_min'] = dataset.min(axis=1)

        # Calculate the max by row
        dataset['row_max'] = dataset.max(axis=1)

        # Calculate the variance by row
        dataset['row_variance'] = dataset.var(axis=1)

        return dataset

    @classmethod
    def _create_other_features(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        """

        :param dataset:
        :param targets:
        :return:
        """

        # dataset['Column_test_1'] = (
        #     dataset["mean_value_of_short_term_variability"] * dataset["abnormal_short_term_variability"])
        #
        # dataset['Column_test_2'] = (
        #     dataset["mean_value_of_long_term_variability"] * dataset["percentage_of_time_with_abnormal_long_term_variability"])

        dataset['Column_test_3'] = (
            dataset["baseline value"] * dataset["abnormal_short_term_variability"]
        )

        dataset['Column_test_4'] = (
                dataset["baseline value"] * dataset["uterine_contractions"]
        )

        dataset['Column_test_5'] = (
                dataset["baseline value"] * dataset["accelerations"]
        )

        # dataset['Column_test_6'] = (
        #         dataset["baseline value"] * dataset["fetal_movement"]
        # )

        dataset['Column_test_9'] = (
             (dataset['uterine_contractions'] * dataset["histogram_number_of_peaks"])
        )

        # Interaction between "Fetal Movement" and "accelerations"
        dataset['interaction_uterine_fetal'] = dataset['uterine_contractions'] * dataset['fetal_movement']

        # Interaction between "Fetal Movement" and "accelerations"
        dataset['interaction_movement_accelerations'] = dataset['fetal_movement'] * dataset['accelerations']

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

        # Sum of each line
        dataset['sum_rows'] = dataset.iloc[:, :-1].sum(axis=1)

        return dataset

    @classmethod
    def _create_categorical_feature(cls, dataset: pd.DataFrame, new_feature_name: str, column_name: str) -> pd.DataFrame:
        """
            Creates categorical features for column specified by <FeatureEngineering.categories>
            :param column_name: feature name.
            :param new_feature_name: name of the categorical feature created.
            :param dataset: dataset that a new feature will be added
            :return dataset: the dataset with a new feature added
        """

        # Define the bins and labels for the categorical binning
        accelerations_min = dataset[column_name].min()
        accelerations_max = dataset[column_name].max()
        accelerations_interval = (accelerations_max - accelerations_min) / len(FeatureEngineering.categories)

        # Calculating the values that separate the bins
        bins = [accelerations_min]
        while bins[-1] < accelerations_max:
            bins.append(bins[-1] + accelerations_interval)

        print(bins)

        # Perform categorical binning for a specific column, e.g., 'accelerations'
        dataset[new_feature_name] = (
            pd.cut(dataset[column_name], bins=bins, labels=FeatureEngineering.categories,
                   include_lowest=True))

        # Transform categorical to int starting from 1
        dataset[new_feature_name] = (dataset[new_feature_name].cat.codes + 1).astype(int)

        return dataset

    @classmethod
    def create_features(cls, dataset: pd.DataFrame, targets: list) -> pd.DataFrame:
        """
            Creates the default features. Mean,Std,min,max,median ...
            :param dataset: The dataset to be used for adding features
            :param targets: The targets features to place in the end
            :return DataFrame: A DataFrame containing all the features
        """

        # Removing columns to ignore
        dataset_targets = dict()
        for target in targets:
            dataset_targets.update({target: dataset.pop(target)})

        dataset = cls._create_default_features(dataset)
        dataset = cls._create_other_features(dataset)
        dataset = cls._create_categorical_feature(dataset, "accelerations_category", "accelerations")
        dataset = cls._create_categorical_feature(dataset, "fetal_movement_category", "fetal_movement")

        # Adding ignored columns to the end of the dataset
        for key in dataset_targets.keys():
            dataset[key] = dataset_targets[key]

        return dataset
