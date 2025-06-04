# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import itertools

# installed libraries
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import ARDRegression, HuberRegressor, LassoLarsIC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# local files
import get_data
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

FRUITS = ["tomato", "mango"]
MEASUREMENT_TYPES = ["reflectance", "absorbance"]
INT_TIMES = [50, 100, 150, 200, 250]
LED_CURRENTS = ["12.5 mA", "25 mA", "100 mA"]
SCORE = "neg_mean_absolute_error"



regression_models_2_3 = {
    "ARD": ARDRegression(lambda_1=1, lambda_2=1),
    "Huber Regression": HuberRegressor(alpha=0.1, epsilon=5, max_iter=10000),
    "Lasso IC": LassoLarsIC(criterion='bic'),
    "SVR": SVR(epsilon=0.5, kernel="rbf", C=10, gamma=0.1),
    "PLS": PLSRegression(n_components=5),
    "KPCA-SVR": Pipeline([
                        ('kpca', KernelPCA(gamma=0.01, kernel="linear", n_components=6)),
                        ('svr', SVR(C=10, epsilon=0.5, gamma=0.1))
                    ])
}

regression_models_5x = {
    "ARD": ARDRegression(lambda_1=1, lambda_2=1),
    "Huber Regression": HuberRegressor(alpha=0.1, epsilon=5, max_iter=10000),
    "Lasso IC": LassoLarsIC(criterion='bic'),
    "SVR": SVR(epsilon=0.1, kernel="rbf", C=100, gamma=0.1),
    "PLS": PLSRegression(n_components=10),
    "KPCA-SVR": Pipeline([
                        ('kpca', KernelPCA(gamma=0.01, kernel="linear", n_components=10)),
                        ('svr', SVR(C=10, epsilon=0.5, gamma=0.1))
                    ])
}


def make_anova_table_file(sensor: str, cv_repeats: int = 10):
    """
    Generate an ANOVA table CSV file for a given sensor and fruit combinations
    by running Partial Least Squares (PLS) regression models with various
    parameter settings.

    Parameters
    ----------
    sensor : str
        The name of the sensor used for data collection (e.g., 'as7262', 'as7265x').
    cv_repeats : int, optional
        The number of cross-validation splits for StratifiedGroupShuffleSplit
        during model evaluation (default is 10).

    Outputs
    -------
    Saves a CSV file named "ANOVA_<sensor>.csv" in the "ANOVA_data" directory.
    The file includes columns:
        - Leaf: The fruit sample analyzed.
        - Measurement Type: The type of measurement (e.g., 'reflectance', 'absorbance', 'raw').
        - Integration Time: The integration time for the measurement.
        - LED Current: The current applied to the LED (e.g., '12.5 mA', '25 mA').
        - Score: The average cross-validation score for the given parameter combination.
    """
    results = []
    for fruit in FRUITS:
        targets = ("lycopene (FW)", "beta-carotene (FW)")
        if fruit == "mango":
            targets=("carotene (FW)", "phenols (FW)")
        print(f"Making {fruit} {sensor} ANOVA table")

        int_times = INT_TIMES
        cv = StratifiedGroupShuffleSplit(n_splits=cv_repeats, test_size=0.2,
                                         n_bins=10)
        all_regrs = regression_models_2_3
        if sensor == "as7265x":
            int_times = [50, 100, 150]
            all_regrs = regression_models_5x
        elif sensor == "c12880":
            all_regrs = regression_models_5x
        combinations = itertools.product(MEASUREMENT_TYPES, int_times, LED_CURRENTS, all_regrs,
                                         targets)

        for measure_type, int_time, current, regr_key, target in combinations:
            if sensor == "as7265x" and current == "100 mA":  # was not tested
                continue
            print(regr_key)

            x, y, groups = get_data.get_cleaned_data(
                fruit=fruit, sensor=sensor,
                targets=(target, ),
                measurement_mode=measure_type,
                int_time=int_time, led_current=current)

            # Further preprocessing based on the 'preprocess' parameter
            regressor = all_regrs[regr_key]

            # Scaling the data
            x_current = StandardScaler().fit_transform(x)

            # for i in range(repeats):

            # Predict using cross-validation
            # print(y)
            y = y[target]
            scores = cross_val_score(
                regressor, x_current, y, groups=groups, cv=cv,
                scoring=SCORE)
            for score in scores:
                # Append the results
                results.append({
                    "Fruit": fruit,
                    "Target": target,
                    "Regressor": regr_key,
                    "Measurement Type": measure_type,
                    "Integration Time": int_time,
                    "LED Current": current,
                    "Score": score
                })
            print(len(results))

    # Create a DataFrame from the results and save it to a CSV file
    results_df = pd.DataFrame(results)
    filename = f"ANOVA_data/ANOVA_{sensor}.csv"
    results_df.to_csv(filename, index=False)

if __name__ == '__main__':
    for sensor_ in ["as7262", "as7263", "as7265x", "c12880"]:
        make_anova_table_file(sensor_)
    # make_anova_table_file("as7262")
    # print(get_data.get_targets("mango"))
