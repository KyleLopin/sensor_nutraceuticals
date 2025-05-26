# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import itertools

# installed libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# sklearn models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ARDRegression, HuberRegressor, Lasso

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
LED_CURRENTS = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
INT_TIMES = [50, 100, 150, 200, 250]
MEASUREMENT_TYPES = ["reflectance", "absorbance"]
FRUITS = ["mango", "tomato"]
CV = GroupShuffleSplit(test_size=0.2, n_splits=10)


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

        # Cross-validate using the provided CV strategy
        # print(groups)
        for train_idx, test_idx in cv.split(x, y, groups=groups["Fruit"]):
            # Use standard indexing for numpy arrays
            x_train, x_test = x[train_idx], x[test_idx]
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


def plot_4_sensors_pls(fruit: str):
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
            groups = groups[["Fruit"]]
            # print(x, y)
            # print(get_data.get_targets(fruit))
            # 
            print("====")
            print(groups)
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


def make_grid_searches(regr_type:str, sensors: list=[], show_figures=False):
    if not show_figures:
        matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
    if not sensors:
        sensors = ["as7262", "as7263", "as7265x", "c12880"]
    for sensor in sensors:
        led = "White LED"
        if sensor =="as7265x":
            led = "b'White IR'"
        pdf_filename = f"{regr_type} grid wide search {sensor}.pdf"
        with PdfPages(pdf_filename) as pdf:
            combinations = itertools.product(FRUITS, MEASUREMENT_TYPES, INT_TIMES, LED_CURRENTS)
            for fruit, measure_type, int_time, current in combinations:
                if sensor == "as7265x" and int_time in [200, 250]:
                    continue
                if sensor == "c12880":
                    if (int_time in [100, 150, 200, 250] or
                            current in ["25 mA", "50 mA", "100 mA"]):
                        continue
                target = "lycopene (FW)"
                if fruit == "mango":
                    target = "carotene (FW)"
                x, y, groups = get_data.get_cleaned_data(
                    sensor=sensor, fruit=fruit,
                    targets=[target],
                    measurement_mode=measure_type,
                    int_time=int_time, led_current=current, led=led)

                scaler = StandardScaler()
                print("a: ", sensor, fruit, measure_type)
                x = pd.DataFrame(
                    scaler.fit_transform(x),
                    columns=x.columns,
                    index=x.index
                )
                y = y.iloc[:, 0]  # supress errors
                groups = groups["Fruit"]
                title = f"leaf: {fruit}, {measure_type}, int time: {int_time}, current: {current}"
                if regr_type == "Huber":
                    param_grid = {
                        'epsilon': [1, 1.35, 1.55, 1.7, 2.0, 2.5, 3, 3.5, 4,
                                    5, 6, 7, 8, 9, 10],
                        'alpha': np.logspace(-5, 2, num=10),
                    }

                    # Initialize the Huber Regressor
                    regr = HuberRegressor(max_iter=1000)
                elif regr_type == "ARD":
                    param_grid = {
                        'alpha_1': np.logspace(-6, -2, num=10),
                        'alpha_2': np.logspace(-6, -2, num=10),
                        'lambda_1': np.logspace(-6, 0, num=10),
                        'lambda_2': np.logspace(-6, 0, num=10),
                    }
                    regr = ARDRegression()
                elif regr_type == "Ridge":
                    param_grid = {
                        'alpha': np.logspace(-5, 3, num=10),
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
                    }
                    from sklearn.linear_model import Ridge
                    regr = Ridge(max_iter=1000)

                elif regr_type == "Lasso":
                    param_grid = {
                        'alpha': np.logspace(-5, 1, num=10),
                        'max_iter': [1000, 5000],
                    }
                    from sklearn.linear_model import Lasso
                    regr = Lasso()

                elif regr_type == "ElasticNet":
                    param_grid = {
                        'alpha': np.logspace(-5, 1, num=8),
                        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                        'max_iter': [1000],
                    }
                    from sklearn.linear_model import ElasticNet
                    regr = ElasticNet()

                elif regr_type == "SVR":
                    param_grid = {
                        'C': np.logspace(-2, 3, 10),
                        'gamma': np.logspace(-4, 0, 5),
                        'epsilon': [0.1, 0.2, 0.5, 1.0],
                        'kernel': ['rbf'],
                    }
                    from sklearn.svm import SVR
                    regr = SVR()
                elif regr_type == "DecisionTree":
                    from sklearn.tree import DecisionTreeRegressor
                    param_grid = {
                        'max_depth': [3, 5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 5]
                    }
                    regr = DecisionTreeRegressor()

                elif regr_type == "RandomForest":
                    from sklearn.ensemble import RandomForestRegressor
                    param_grid = {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10, None],
                        'max_features': ['sqrt', 'log2'],
                        'min_samples_split': [2, 5]
                    }
                    regr = RandomForestRegressor(n_jobs=-1)

                elif regr_type == "ExtraTrees":
                    from sklearn.ensemble import ExtraTreesRegressor
                    param_grid = {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5]
                    }
                    regr = ExtraTreesRegressor(n_jobs=-1)

                elif regr_type == "GradientBoosting":
                    from sklearn.ensemble import GradientBoostingRegressor
                    param_grid = {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5],
                        'min_samples_split': [2, 5]
                    }
                    regr = GradientBoostingRegressor()

                elif regr_type == "HistGradientBoosting":
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    param_grid = {
                        'learning_rate': [0.01, 0.1],
                        'max_iter': [100, 200],
                        'max_leaf_nodes': [15, 31, 63],
                        'min_samples_leaf': [20, 50]
                    }
                    regr = HistGradientBoostingRegressor()


                elif regr_type == "AdaBoost":
                    from sklearn.ensemble import AdaBoostRegressor
                    from sklearn.tree import DecisionTreeRegressor

                    param_grid = {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.01, 0.1, 1.0],
                        'estimator__max_depth': [2, 3, 5]
                    }
                    base_tree = DecisionTreeRegressor()
                    regr = AdaBoostRegressor(estimator=base_tree)

                elif regr_type == "KNN":
                    from sklearn.neighbors import KNeighborsRegressor
                    param_grid = {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
                    }
                    regr = KNeighborsRegressor()
                elif regr_type == "BayesianRidge":
                    from sklearn.linear_model import BayesianRidge
                    param_grid = {
                        'alpha_1': np.logspace(-6, -2, 4),
                        'alpha_2': np.logspace(-6, -2, 4),
                        'lambda_1': np.logspace(-6, -2, 4),
                        'lambda_2': np.logspace(-6, -2, 4)
                    }
                    regr = BayesianRidge()
                elif regr_type == "KPCA_LR":
                    from sklearn.decomposition import KernelPCA
                    from sklearn.linear_model import LinearRegression
                    from sklearn.pipeline import Pipeline
                    from sklearn.model_selection import GridSearchCV

                    param_grid = {
                        'kpca__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'kpca__n_components': [3, 5, 6]
                    }

                    # Use a pipeline: KernelPCA → LinearRegression
                    regr = Pipeline([
                        ('kpca', KernelPCA(fit_inverse_transform=False, eigen_solver='auto')),
                        ('lr', LinearRegression())
                    ])

                else:
                    raise ValueError(f"Unsupported regression type: {regr_type}")

                make_regr_grid_search_best_params(
                    regr, param_grid, x, y, groups, title, pdf)


def make_regr_grid_search_best_params(
        regr, param_grid: dict, x: pd.DataFrame, y: pd.Series,
        groups: pd.Series, title: str,
        pdf: PdfPages, show_figure: bool = False):

    # Perform grid search
    grid_search = GridSearchCV(
        regr, param_grid, cv=CV, scoring='r2', n_jobs=8)
    # print(x)
    # print(y)
    # print(groups)
    grid_search.fit(x, y, groups=groups)
    # Extract results
    results = grid_search.cv_results_

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
   # Initialize a dictionary to store the best scores for each hyperparameter
    best_scores = {key: [] for key in param_grid}
    # Extract best scores for each hyperparameter value
    for param in best_scores.keys():
        for value in param_grid[param]:
            mask = results_df[f'param_{param}'] == value
            best_score = results_df[mask]['mean_test_score'].max()
            best_scores[param].append((value, best_score))

    # Convert best scores to a DataFrame for plotting
    best_scores_df = {param: pd.DataFrame(
        scores, columns=[param, 'best_score'])
        for param, scores in best_scores.items()}
    plt.figure(figsize=(10, 6))
    # Plot each hyperparameter's best score
    for param, df in best_scores_df.items():
        plt.plot(df[param], df['best_score'], label=param, marker='o')
        plt.xscale('log')  # Set the x-axis to logarithmic scale

    plt.xlabel('Hyperparameter Value')
    plt.ylabel('Best R2 Score')
    plt.title(f'Best R2 Score for Each Hyperparameter Value (Log Scale)\n{title}')
    plt.legend()
    plt.grid(True)
    # Best parameters and model
    best_params = grid_search.best_params_
    print("Best parameters found:", best_params)
    best_params_text = "\n".join([f"{key}: {value}"
                                  for key, value in best_params.items()])
    best_score = grid_search.best_score_
    print(f"Best R² Score: {best_score:.4f}")
    plt.annotate(f'Best Parameters:\n{best_params_text}', xy=(0.05, 0.95),
                 xycoords='axes fraction',
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3",
                           edgecolor="black", facecolor="white"))
    if show_figure:
        plt.show()
    else:
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    # pls_scan("tomato", "as7265x", "lycopene (DW)")
    # print(get_data.get_targets("tomato"))
    # for target in ['%DM', 'lycopene (DW)', 'lycopene (FW)', 'beta-carotene (DW)', 'beta-carotene (FW)']:
    #     plot_3_sensors("tomato", target)
    # plot_4_sensors_pls("tomato")
    # ["DecisionTree", "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "ARD"]
    for regr in ["AdaBoost"]:
        make_grid_searches(regr)
