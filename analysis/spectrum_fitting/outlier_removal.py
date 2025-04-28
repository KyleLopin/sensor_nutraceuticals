# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet

# local files
import get_data


def calculate_residues(x: pd.DataFrame,
                       groups: pd.Series) -> pd.DataFrame:
    """
    Calculate residuals for each row in a DataFrame based on the difference of the individual
    value(s) and the average of all the other rows in that group.

    Args:
        x (pandas.DataFrame): Input DataFrame containing the data.
        groups (pandas.Series): Series defining the groups for each row in the DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing residuals for each row,
        calculated based on group-wise means.
        The index and columns of the returned DataFrame match those of the input DataFrame 'x'.
    """
    if isinstance(x, pd.DataFrame):
        new_df = pd.DataFrame(columns=x.columns)
    elif isinstance(x, pd.Series):
        new_df = pd.Series()
    for index in x.index:
        # get indexes with the same leaf number minus the current index
        # print(index)
        # print(groups)
        # print(groups.loc[index])
        fruit, spot, read_number = groups.loc[index]
        # other_leaf_indexes = groups.index[groups == groups[index]].drop(index)
        # Find all other rows with same fruit and spot (but different read number)
        mask = ((groups["Fruit"] == fruit) &
                (groups["spot"] == spot))
        # calculate the difference between the current index and the mean of the
        # spectrum for the other reads on same fruit / spot
        other_spot_indexes = groups.index[mask].drop(index)
        residue = x.loc[index] - x.loc[other_spot_indexes].mean()
        # make new DataFrame with the residues
        new_df.loc[index] = residue
    return new_df


def mahalanobis_outlier_removal(x: pd.DataFrame,
                                use_robust: bool = True,
                                display_hist: bool = False,
                                cutoff_limit: float = 4.0,
                                ) -> np.ndarray[bool]:
    """
    Remove outliers from a DataFrame based on Mahalanobis distance.

    Parameters:
        x (pd.DataFrame): Input DataFrame containing numerical data.
        use_robust (bool): Whether to use a robust estimator (default True).
        display_hist (bool): Whether to display histograms (default False).
        If True the function will also show the data in a blocking way.
        cutoff_limit (float): Number of standard deviations to consider outliers (default 3.0).

    Returns:
        pd.DataFrame: DataFrame mask
    """
    if use_robust:  # fit a MCD robust estimator to data
        covariance = MinCovDet()
    else:  # fit a MLE estimator to data
        covariance = EmpiricalCovariance().fit(x)
    covariance.fit(x)
    # calculate the mahalanobis distances
    # and calculate the cubic root to simulate a normal distribution
    mahal_distances = covariance.mahalanobis(x - covariance.location_)**0.33
    # shift the distribution to calculate the distance from the standard deviation easier
    shifted_mahal = mahal_distances - mahal_distances.mean()
    cutoff = cutoff_limit*shifted_mahal.std()
    data_mask = np.where((-cutoff < shifted_mahal) & (shifted_mahal < cutoff), True, False)
    if display_hist:
        plt.hist(shifted_mahal, bins=100)
        plt.axvline(cutoff, ls='--')
        print(f"outliers removed = {x.shape[0]-x[data_mask].shape[0]}")
        plt.show()
    print(f"outliers removed = {x.shape[0] - x[data_mask].shape[0]}")
    return data_mask


if __name__ == '__main__':
    sensor = "c12880"
    fruit = "tomato"
    measurement_type = "reflectance"
    # led = "White LED"
    sensor_settings = {"led current": "12.5 mA",
                       "integration time": 50}
    if sensor == "as7265x":
        sensor_settings["led"] = "b'White IR'"
    x, y, groups = get_data.get_data(sensor=sensor,
                                     fruit=fruit,
                                     measurement_mode=measurement_type,
                                     mean_spot=False,
                                     target_column='lycopene (FW)',
                                     **sensor_settings)
    residues = calculate_residues(x, groups)
    mahalanobis_outlier_removal(residues, display_hist=True)
