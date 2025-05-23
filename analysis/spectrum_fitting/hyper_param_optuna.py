# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Use Optuna hyperparameter tuning visualization tools
"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import numpy as np
import optuna
import optuna.visualization
import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import GroupKFold, cross_val_score

# local files
import get_data


def optimize_ard_with_optuna(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 50,
    direction: str = "maximize",
    random_state: int = 42,
) -> optuna.study.Study:
    """
    Run Optuna hyperparameter tuning for ARDRegression using group-aware CV.

    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels for cross-validation
        n_trials: Number of Optuna trials
        direction: "maximize" (e.g. R²) or "minimize" (e.g. MSE)
        random_state: Seed for reproducibility

    Returns:
        Optuna Study object
    """
    def objective(trial: optuna.Trial) -> float:
        alpha_1 = trial.suggest_loguniform("alpha_1", 1e-6, 1e-2)
        alpha_2 = trial.suggest_loguniform("alpha_2", 1e-6, 1e-2)
        lambda_1 = trial.suggest_loguniform("lambda_1", 1e-6, 1e-2)
        lambda_2 = trial.suggest_loguniform("lambda_2", 1e-6, 1e-2)

        model = ARDRegression(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2
        )

        cv = GroupKFold(n_splits=5)
        print(X)
        print(y)
        print(groups)
        score = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2").mean()
        return score

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    # Visualizations
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_contour(study).show()

    return study


if __name__ == '__main__':
    sensor = "as7265x"
    y_columns = "lycopene (FW)"
    fruit = "tomato"
    led = "b'White IR'"
    x, y, groups = get_data.get_cleaned_data(
        sensor=sensor, int_time=50, targets=[y_columns],
        led_current="12.5 mA", fruit=fruit,
        measurement_mode="absorbance", led=led)
    print(x, y, groups)
    results = optimize_ard_with_optuna(x, y, groups["Fruit"])
