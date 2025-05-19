# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

# used to choose the number of components from the AIC score, and this
# will be repeated an additional N_SPLITS_OUTER for final AIC and R2 scores
MIN_COMPONENTS = 3
N_SPLITS = 100  # To reduce variance in final score use high numbers
RANDOM_STATE = 43
USE_RFECV = False
USE_SFS = False


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


def pls_scan(ax: plt.Axes, x: pd.DataFrame, y: pd.Series, groups: pd.DataFrame, max_comps):

    # print(x, y)
    # Standardize data before Mahalanobis distance
    # poly = PolynomialFeatures(degree=2)
    # x = poly.fit_transform(x)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # print("x: ", x)
    # print(y)
    # print("nan: ", x[np.isnan(x).any(axis=1)])
    results = grid_search_pls_aic_r2(pd.DataFrame(x), y, groups,
                                     max_components=max_comps)
    # print(results)
    # for key, value in results.items():
    #     print(f"{key}: {value}")
    ax.plot(results["n_components"], results["mean_r2"])
    ax.plot(results["n_components"], results["r2_train"])
    ax.fill_between(results["n_components"],
                    np.array(results["mean_r2"]) - np.array(results["std_r2"]),
                    np.array(results["mean_r2"]) + np.array(results["std_r2"]),
                    alpha=0.3)


def grid_search_pls_aic_r2(x, y, groups, cv=None, max_components: int = 6):
    """
    Perform a cross-validated grid search to find the optimal number of components
    for Partial Least Squares (PLS) regression using Akaike Information Criterion (AIC)
    and R^2 score.

    Args:
        x (pd.DataFrame or np.ndarray): Feature matrix containing the predictor variables.
        y (pd.Series or np.ndarray): Target variable values.
        groups (pd.Series or np.ndarray): Group labels for the samples used for cross-validation.
                                          This allows splitting data based on group information.
        cv (cross-validation generator, optional): Cross-validation splitting strategy.
                                                   Defaults to `GroupShuffleSplit` with 20 splits
                                                   and a test size of 0.25.
        max_components (int, optional): The maximum number of PLS components to test.
                                        Defaults to 6.

    Returns:
        dict: A dictionary containing lists of AIC and R^2 statistics
              for each number of components.
    """
    # Convert x to numpy array for consistent indexing
    x = np.array(x)
    y = np.array(y)

    if not cv:
        cv = GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.25)
        # cv = StratifiedGroupShuffleSplit(n_splits=N_SPLITS_INNER, test_size=0.25,
        #                                  n_bins=10, random_state=RANDOM_STATE)

    # Initialize a dictionary to store results as lists
    results = {
        'n_components': [],
        'mean_aic': [],
        'std_aic': [],
        'mean_r2': [],
        'std_r2': [],
        'mean_mae': [],
        'std_mae': [],
        "r2_train": []
    }

    num_samples = x.shape[0]

    # Loop through each number of components to calculate AIC and R^2 scores
    for n_components in range(MIN_COMPONENTS, max_components + 1):
        pls = PLSRegression(n_components=n_components, max_iter=10000)
        aic_list = []
        r2_list = []
        r2_train = []
        mae_list = []

        if USE_RFECV:
            # Apply RFECV to select features within each CV split
            rfecv = RFECV(
                estimator=pls,
                step=1,
                cv=cv,  # Small CV within each fold to determine features
                scoring='neg_mean_absolute_error',
                min_features_to_select=max_components,
                n_jobs=-1
            )
            rfecv.fit(x, y, groups=groups)

            # print(f"optimal number of features {rfecv.n_features_}")

            # Use the selected features from RFECV for PLS
            x_fs = rfecv.transform(x)
        elif USE_SFS:
            sfs = SequentialFeatureSelector(
                estimator=pls,
                tol=0.005,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            x_fs = sfs.fit_transform(x, y, groups=groups)
        else:
            x_fs = x

        # Cross-validate using the provided CV strategy
        # print(groups)
        for train_idx, test_idx in cv.split(x_fs, y, groups=groups["Fruit"]):
            # Use standard indexing for numpy arrays
            x_train, x_test = x_fs[train_idx], x_fs[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # print("====")
            # print(x_train.shape, x_test.shape)
            # Fit the model
            pls.fit(x_train, y_train)
            y_pred = pls.predict(x_test)
            y_pred_train = pls.predict(x_train)
            # Calculate Residual Sum of Squares (RSS)
            rss = np.sum((y_test - y_pred) ** 2)

            # Calculate AIC for this fold
            num_parameters = n_components + 1  # Components + intercept
            aic = calculate_aic(rss, num_samples, num_parameters)
            aic_list.append(aic)

            # Calculate R^2 score for this fold
            # r2 = r2_score(y_test, y_pred)
            # to calculate the r2 score after averaging the groups
            y_pred_grouped = pd.Series(y_pred.ravel(), index=groups.iloc[test_idx]["Fruit"]
                                       ).groupby('Fruit').mean()
            y_test_grouped = pd.Series(y_test, index=groups.iloc[test_idx]["Fruit"]
                                       ).groupby('Fruit').mean()
            y_train_grouped = pd.Series(y_train, index=groups.iloc[train_idx]["Fruit"]
                                        ).groupby('Fruit').mean()
            y_train_pred = pd.Series(y_pred_train.ravel(), index=groups.iloc[train_idx]["Fruit"]
                                     ).groupby('Fruit').mean()
            r2 = r2_score(y_test_grouped, y_pred_grouped)
            r2_list.append(r2)
            r2_train.append(r2_score(y_train_grouped, y_train_pred))
            mae_list.append(mean_absolute_error(y_test, y_pred))

        # Compute the mean and standard deviation of AIC and R^2 across CV folds and
        # Store AIC and R^2 statistics for current number of components
        results['n_components'].append(n_components)
        results['mean_aic'].append(np.mean(aic_list))
        results['std_aic'].append(np.std(aic_list))
        results['mean_r2'].append(np.mean(r2_list))
        results['std_r2'].append(np.std(r2_list))
        results['mean_mae'].append(np.mean(mae_list))
        results['std_mae'].append(np.std(mae_list))
        results["r2_train"].append(np.mean(r2_train))

    return results


def plot_4_sensors(fruit: str):
    width = 5

    y_columns = tuple(get_data.get_targets(fruit))
    # just test free weight targets
    y_columns = tuple([x for x in y_columns if " (FW)" in x])
    print(y_columns)
    for target in y_columns:
        figure, axes = plt.subplots(4, 1, figsize=(width, 1.3 * width),
                                    sharex=True)
        figure.suptitle(target)
        for i, (sensor, ax) in enumerate(zip(["as7262", "as7263", "as7265x", "c12880"], axes)):
            led = "White LED"
            max_comps = 6
            # more max components for sensors with more wavelengths
            if sensor in ["as7265x", "c12880"]:
                led = "b'White IR'"
                max_comps = 14

            # get all coluns 1 time, get_data.get_data is caching the data, this way on
            # read each csv once
            x, y, groups = get_data.get_cleaned_data(
                sensor=sensor, int_time=50, targets=y_columns,
                led_current="12.5 mA", fruit=fruit,
                measurement_mode="absorbance", led=led)
            # print(x, y)
            # print(get_data.get_targets(fruit))
            # 
            # print("====")
            # print(y)
            # print(target)
            # print(y[target])
            y_target = y[target].to_numpy().flatten()
            print("==sensor==", sensor)
            pls_scan(ax, x, y_target, groups, max_comps)
            ax.set_xlabel("number of components")
            ax.set_ylabel("$R^2$")
            if plt.ylim()[0] < 0:
                plt.ylim(bottom=0)
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # pls_scan("tomato", "as7265x", "lycopene (DW)")
    # print(get_data.get_targets("tomato"))
    # for target in ['%DM', 'lycopene (DW)', 'lycopene (FW)', 'beta-carotene (DW)', 'beta-carotene (FW)']:
    #     plot_3_sensors("tomato", target)
    plot_4_sensors("tomato")
