import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed


class ModelSelection:
    """
        Classe responsible for showing up the data, making it easier to analyze data sparsity, distribution, ... TODO: add docstring
    """
    def FeatureAdd(self, file="fetal_health.csv"):
        # Sample DataFrame
        data = pd.read_csv(file)

        df = pd.DataFrame(data)
        time_window = 3
        # Interaction between "Uterine Contractions" and "Fetal Movement,"
        df['interaction_uterine_fetal'] = df['uterine_contractions'] * df['fetal_movement']

        df['rolling_mean_accelerations'] = df['accelerations'].rolling(window=time_window).mean()
        df['rolling_std_decels'] = df['severe_decelerations'].rolling(window=time_window).std()
        # Calculate the mean by row
        df['row_mean'] = df.mean(axis=1)
        # Calculate the median by row
        df['row_median'] = df.median(axis=1)
        # Calculate the min by row
        df['row_min'] = df.min(axis=1)
        # Calculate the max by row
        df['row_max'] = df.max(axis=1)
        # Calculate the variance by row
        df['row_variance'] = df.var(axis=1)
        # Calculate the mode by row
        row_modes = df.mode(axis=1)
        # Extract the first mode for each row
        df['row_mode'] = row_modes.iloc[:, 0]
        # Calculate the Skewness by row
        # If skewness is greater than 0, it indicates that the right
        # tail of the distribution is longer or fatter than the left tail. If it lower
        # than 0 it indicates the oposite
        df['row_skewness'] = df.skew(axis=1)
        # Define the bins and labels for the categorical binning
        bins = [0, 50, 100, 150, float('inf')]  # Adjust the bin edges as needed
        labels = ['Low', 'Medium', 'High', 'Very High']
        # Perform categorical binning for a specific column, e.g., 'accelerations'
        df['accelerations_category'] = pd.cut(df['accelerations'], bins=bins, labels=labels, include_lowest=True)

        # Display the DataFrame with the new 'bmi' feature
        print(df)




