# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data


def calculate_aic(residual_sum_of_squares, num_observations, num_parameters):
    """
    Calculate the Akaike Information Criterion (AIC) for a regression model.

    Parameters
    ----------
    residual_sum_of_squares : float
        The residual sum of squares (RSS) from the model, which measures the
        difference between observed and predicted values.

    num_observations : int
        The total number of observations (data points) used in the model.

    num_parameters : int
        The number of estimated parameters in the model, including the intercept.

    Returns
    -------
    aic : float
        The calculated AIC value, which incorporates model fit and complexity.
        A lower AIC value suggests a better model, balancing fit with simplicity.

    Notes
    -----
    The formula for AIC is:

        AIC = n * log(RSS / n) + 2 * k

    where:
        - n is the number of observations (num_observations)
        - RSS is the residual sum of squares
        - k is the number of parameters (num_parameters)
        - log is the natural logarithm

    A lower AIC indicates a model that better balances fit and simplicity.

    References
    ----------
    Akaike, Hirotugu. "A new look at the statistical model identification."
    *IEEE Transactions on Automatic Control* 19, no. 6 (1974): 716-723.
    doi:10.1109/TAC.1974.1100705
    """
    return num_observations * np.log(residual_sum_of_squares / num_observations) + \
        2 * num_parameters


def pls_scan(fruit: str, sensor: str):
    led = "White LED"
    max_comps = 6
    if sensor == "as7265x":
        led = "b'White IR'"
        max_comps = 14
    y_columns = ('%DM', 'lycopene (DW)', 'lycopene (FW)',
                                                    'beta-carotene (DW)', 'beta-carotene (FW)')
    x, y, groups = get_data.get_data(sensor=sensor, int_time=50,
                                     target_column=y_columns,
                                     led_current="12.5 mA", fruit=fruit,
                                     measurement_mode="absorbance",
                                     led=led, mean_spot=True)
    print(x, y)
    print(get_data.get_targets(fruit))
    # Standardize data before Mahalanobis distance
    scaler = StandardScaler()
    x = scaler.fit_transform(x)



    plt.plot(x.T)
    plt.show()


if __name__ == '__main__':
    pls_scan("tomato", "as7265x")
