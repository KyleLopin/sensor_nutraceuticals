# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Load spectral data sets for tomato and mango fruit.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from functools import lru_cache
from pathlib import Path

# installed libraries
import numpy as np
import pandas as pd

DATA_FOLDER = Path(__file__).parent.parent.parent / "data"
pd.set_option('display.max_rows', 10)  # noqa
pd.set_option('display.width', None)  # noqa


def is_float(value):
    """
    Check if a value can be converted to a float.

    Parameters:
        value: The input value to check (typically a string or number).

    Returns:
        bool: True if the value can be converted to a float, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_raw_c12880_data(fruit: str = "tomato",
                        data_type: str = "data",
                        wavelength_range: tuple[float, float] = (412, 691),
                        dark_current_cutoff: float = None) -> pd.DataFrame:
    """
    Load raw spectrometer from C12880 on tomatoes and mangos.

    This function reads a dataset for a given fruit and filters the data based
    on the specified `data_type`.

    Parameters
    ----------
    fruit : str, optional
        The name of the fruit whose dataset is to be processed, "tomato" or "mango".

    data_type : str, optional
        The type of data to retrieve. Must be one of:
        - `"reference"` : Returns only reference measurements of white background.
        - `"dark"` : Returns only dark calibration measurements.
        - `"data"` : Returns spectral data from fruit.
        Default is `"data"`.

    wavelength_range : list[float]
        Specifies the range of wavelengths (inclusive) to keep in the dataset.
        Default is `[412, 619]`. If an empty list `[]` is provided, the full spectrum is retained.

    dark_current_cutoff : float | None, optional
        Wavelength threshold to identify dark current columns. If provided,
        wavelengths below this value are averaged and subtracted from the rest.

    Returns
    -------
    pd.DataFrame
        A processed DataFrame containing the relevant spectral data, with:
        - `"sample"` : Sample numbers
        - `"spot"` : Spot number ont he sample.
        - Remaining columns : Spectral intensity values at different wavelengths
          (with dark current correction if applicable).

    Raises
    ------
    ValueError
        If `data_type` is not one of `"reference"`, `"dark"`, or `"data"`.

    Notes
    -----
    - The `sample` column is forward-filled because it was not fully entered during data collection.
    - Wavelengths outside the specified `wavelength_range` are removed.
    - Duplicate columns are dropped.
    Example
    -------
    >>> df = get_raw_c12880_data(fruit="tomato", data_type="data")
    >>> print(df.head())
       sample  spot  308.38  776.28  775.73  775.15
    0       1   1.1   456.0   448.0   541.0   540.0
    1       1   1.2   452.0   446.0   541.0   541.0
    2       1   1.3   459.0   451.0   541.0   541.0
    3       1   1.4   454.0   446.0   543.0   541.0
    4       2   2.1   463.0   449.0   540.0   541.0
    """
    data_path = DATA_FOLDER / fruit / f"{fruit.capitalize()}_fulldata.xlsx"
    # the Excel file has 2 top rows merged, remove the first one
    data = pd.read_excel(data_path, skiprows=1)  # noqa

    # then add in the names that were removed with skiprows=1
    data.columns = ["sample", "spot"] + list(data.columns[2:])

    # fill in sample numbers down the column, they were not entered during data collection
    # and fix that read_excels converts ints to floats
    data["sample"] = data["sample"].ffill().astype(int)

    if data_type == "reference":
        data = data[data["spot"] == "White"]
    elif data_type == "dark":
        data = data[data["spot"] == "dark"]
    elif data_type == "data":
        data = data[data["spot"].astype(str).str.contains(r"^\d+\.\d+$", na=False)]
    else:
        raise ValueError(f"data_type = {data_type} is not valid, data_type must be in "
                         f"['reference', 'dark', 'data']")

    # because the sensor has to be programmed with wavelengths, some numbers are repeated
    # remove the invalid columns that are repeated with .1 at the end
    invalid_columns = [col for col in data.columns
                       if not is_float(col)]
    print(invalid_columns)
    invalid_columns.remove("spot")
    invalid_columns.remove("sample")
    data = data.drop(columns=invalid_columns)

    spectral_columns = [col for col
                        in data.columns
                        if isinstance(col, (int, float))]

    # Apply dark current subtraction if cutoff is provided
    if dark_current_cutoff is not None:
        # get the dark current, take mean of every wavelength
        # less than the dark current cutoff
        dark_current_cols = [wl for wl in spectral_columns
                             if wl < dark_current_cutoff]

        # Compute a **single** dark current value per sample (mean across all dark current columns)
        dark_current_levels = data[dark_current_cols].mean(axis=1)
        # print(dark_current_levels)

        # Subtract the sample-specific dark current from the corresponding data rows
        data[spectral_columns] = data[spectral_columns].sub(dark_current_levels, axis=0)

    # Apply wavelength range filter
    if wavelength_range:
        filtered_columns = [
            wl for wl in data.columns
            if isinstance(wl, (int, float)) and wavelength_range[0] < wl < wavelength_range[1]
        ]
        data = data[["sample", "spot"]+filtered_columns]
    return data


def _get_c12880_data(fruit: str = "tomato",
                     measurement_mode: str = "reflectance",
                     mean_spot: bool = False,
                     target: str = "lycopene (DW)",
                     wavelength_range: tuple[float, float] = (412, 691),
                     **kwargs):
    """
    Load and process spectral data from the C12880 sensor.

    Parameters:
        fruit (str): Name of the fruit to load data for (used to locate files).
        measurement_mode (str): Either "raw" to use unprocessed sensor data, or
                                "reflectance" to use preprocessed reflectance data.
        mean_spot (bool): Whether to average all spectra from the same spot.
        target (str): Target column name from the AS7262 sensor to use as y-label.
        wavelength_range (tuple): Range of wavelengths to include in the x data.
        **kwargs: Additional keyword arguments passed to the raw data loader.

    Returns:
        x (pd.DataFrame): Spectral data with wavelengths as columns.
        y (pd.Series): Target values aligned with the x data.
        groups (pd.DataFrame): Metadata (e.g., fruit, spot) for grouping samples.
    """
    if measurement_mode == "raw":  # NOT WORKING
        # Load raw sensor data using a separate function
        data = get_raw_c12880_data(fruit=fruit,
                                   data_type="data",
                                   **kwargs)
        raise ValueError("Raw values not working for C12880 now")
        # data_folder = DATA_FOLDER / fruit / measurement_mode
    elif measurement_mode == "reflectance":
        # Load reflectance data from a CSV file
        data_folder = DATA_FOLDER / fruit / "reflectance"
        data_file = data_folder / f"{fruit}_reflectance.csv"
        data = pd.read_csv(data_file)
    else:
        raise ValueError(f"measurement_mode needs to be in ['raw', 'reflectance'], "
                         f"{measurement_mode} is not valid")

    # Extract numeric spot group ID from spot column
    # data["spot_group"] = data["spot"].astype(str).str.split(".").str[0].astype(int)
    agg_dict = {col: 'mean' for col in data.columns}
    agg_dict["sample"] = "first"
    agg_dict["spot"] = "first"
    if mean_spot:
        # Don't need "Read number" as this means nothing after averaging
        # THIS line is not needed after converting files
        # data["spot"] = data["spot"].astype(str).str.extract(r"(\d+)\.").astype(int)
        spectral_cols = [col for col in data.columns if is_float(col)]
        data.rename(columns={"sample": "Fruit"}, inplace=True)
        data = data.groupby(by=["Fruit", "spot"])[spectral_cols].mean()
        # print(data)
        groups = data.index.to_frame(index=True)
    else:  #TODO: figure out the problem here
        # data[["spot", "Read number"]] = (
        #     data["spot"].astype(str).str.extract(r"(\d+)\.(\d+)").astype(int))
        # print('+++++')
        # print(data)
        groups = data[['sample', 'spot', 'Read number']].rename(columns={"sample": "Fruit"})

    invalid_columns = [col for col in data.columns
                       if not is_float(col)]
    x = data.drop(columns=invalid_columns)
    x.columns = [float(col) for col in x.columns]
    if wavelength_range:
        filtered_columns = [
            wl for wl in x.columns
            if isinstance(wl, (int, float)) and wavelength_range[0] < wl < wavelength_range[1]
        ]
        # print("fc: ", filtered_columns, x.columns)
        x = x[filtered_columns]
    # print(x)
    # get y from as7262 data
    y_data_df = pd.read_csv(data_folder / f"{fruit}_as7262_ref_data.csv")
    y = y_data_df.groupby(by=["Fruit"]).agg({target: "first"})
    x = x.reset_index()  # reset Fruit and spot to drop later
    print("=======]]]]")
    print(x)  # This is horrible design, but just trying to pickle good data an move on
    print(groups.columns)
    invalid_columns = [col for col in x.columns
                       if not is_float(col)]
    print(invalid_columns)
    x = x.drop(columns=invalid_columns)
    groups = groups.reset_index(drop=True)
    y = y.reset_index()
    df_merge = groups.merge(y.reset_index(), on="Fruit", how="left")
    y = df_merge[target]
    return x, y, groups


@lru_cache()  # cache the data reads, helps make tests much shorter
def get_data(sensor: str,
             measurement_mode: str = "reflectance",
             fruit: str = "tomato",
             target_column: tuple = ("lycopene (DW)", ),
             split_x_y_groups: bool = True,
             mean_spot: bool = False,
             **kwargs):
    """
    Load and preprocess spectral data from fruit measurement datasets.

    Parameters
    ----------
    sensor : str
        The name of the sensor used. Options include 'as7262', 'as7263', 'as7265x', or 'c12880'.

    measurement_mode : str, default="reflectance"
        The type of data to retrieve. Must be one of:
        - "raw": Unprocessed sensor readings
        - "reflectance": Reflectance values normalized to reference
        - "absorbance": Absorbance values computed as -log10(reflectance)

    fruit : str, default="tomato"
        The fruit type associated with the dataset folder (e.g. "mango", "tomato").

    target_column : tuple of str, default=("lycopene (DW)",)
        One or more column names in the dataset that will be used as target variables.

    split_x_y_groups : bool, default=True
        If True, return (X, Y, group) as separate outputs.
        If False, return the full preprocessed DataFrame.

    mean_spot : bool, default=False
        If True, average all readings for each fruit sample across multiple spots.

    **kwargs :
        Additional sensor-specific filters to apply (e.g. "integration time", "led current").
        These must match column names in the CSV.

    Returns
    -------
    tuple or DataFrame
        If split_x_y_groups is True:
            X : pd.DataFrame
                The spectral feature matrix (columns = wavelengths)
            Y : pd.DataFrame
                The target variable(s)
            groups : pd.DataFrame
                Metadata for grouping (e.g. Fruit, spot, Read number)
        If split_x_y_groups is False:
            full_data : pd.DataFrame
                The complete filtered dataset including all columns.

    Raises
    ------
    KeyError
        If any target_column is not found in the dataset.

    ValueError
        If measurement_mode is invalid.

    Notes
    -----
    - If `sensor="c12880"`, this function defers to a separate loader `_get_c12880_data()`.
    - Uses LRU caching to speed up repeated calls with identical arguments.
    """
    if sensor == "c12880":
        return _get_c12880_data(fruit=fruit,
                                measurement_mode=measurement_mode,
                                mean_spot=mean_spot,
                                target=target_column,
                                **kwargs)

    fruit_folder = DATA_FOLDER / fruit
    if measurement_mode == "raw":
        data_folder = fruit_folder / "raw"
        mid = "_"
    elif measurement_mode in ["reflectance", "absorbance"]:
        data_folder = fruit_folder / "reflectance"
        mid = "_ref_"
    else:
        raise ValueError(f"measurement mode; '{measurement_mode}' is not valid\n"
                         "Use 'raw', 'reflectance', or 'absorbance")

    data = pd.read_csv(data_folder / f"{fruit}_{sensor}{mid}data.csv")
    # print(data)

    # get spectral columns
    x_columns = []
    for column in data.columns:
        if 'nm' in column:
            x_columns.append(column)
    if isinstance(target_column, str):
        # print("converting: ", type(target_column), target_column)
        target_column = (target_column, )
    #     print("converting: ", type(target_column), target_column)
    # print([x for x in target_column])

    missing_columns = [col for col in target_column if col not in data.columns]
    # for col in target_column:
    #     print("+: ", col in data.columns)
    # print(data.columns)
    # print('===')
    # print(missing_columns)
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")
    if kwargs:
        # print("got sensor settings:", kwargs)
        # print(data.columns)
        for sensor_setting, sensor_value in kwargs.items():
            if sensor_setting == "int_time":  # fix this change from last project
                sensor_setting = "integration time"
            if sensor_setting == "led_current":
                sensor_setting = "led current"
            data = data.loc[data[sensor_setting] == sensor_value]

    if mean_spot:
        agg_dict = {col: "mean" for col in x_columns}
        for new_col in ["Fruit", "Fruit number", "spot"] + list(target_column):
            agg_dict[new_col] = "first"
        # print(data.columns)
        # print(agg_dict)
        data = data.groupby(by=["Fruit number"]).agg(agg_dict)
        # print("mean data")
        # print(data)

    x = data[x_columns]
    if measurement_mode == "absorbance":
        print("abs")
        print(x)
        x = -np.log10(x)

    if split_x_y_groups:
        # print(data.columns)
        if mean_spot:
            return x, data[list(target_column)], data[["Fruit", "spot"]]
        else:
            return x, data[list(target_column)], data[["Fruit", "spot", "Read number"]]
    return data


def get_targets(fruit: str) -> list[str]:
    """
    Retrieve target columns related to fruit composition from AS7262 sensor data by
    extracting columns  related to fresh weight (FW),
    dry weight (DW), or percent dry matter (%DM).

    Parameters:
        fruit (str): Name of the fruit to load data for (e.g., "tomato", "mango").

    Returns:
        list[str]: A list of column names corresponding to target variables.

    Notes:
        - Currently matches columns containing "(FW)", "(DW)", or "%DM".
    """
    data = pd.read_csv(DATA_FOLDER / fruit / "raw" / f"{fruit}_as7262_data.csv")
    targets = []
    for column in data.columns:
        if "FW" in column or "(DW)" in column:
            targets.append(column)
        if "%DM" in column:  # is in both, but if add fruits later who knows
            targets.append(column)
    return targets


if __name__ == '__main__':
    # print(get_targets("tomato"))
    # import matplotlib.pyplot as plt
    sensor_settings = {"led current": "12.5 mA",
                       "integration time": 50}
    _x, _y, _groups = get_data("c12880", fruit="mango",
                               measurement_mode="reflectance",
                               mean_spot=True,
                               target_column="%DM",
                               split_x_y_groups=True,
                               **sensor_settings)
    print(_x)
    print(_y)
    # plt.plot(_x.T)
    # plt.show()
    # _data = get_raw_c12880_data(data_type="data",
    #                             dark_current_cutoff=360,
    #                             wavelength_range=None)
    # plt.plot(_data.drop(columns=["spot", "sample"]).T)
    # plt.show()
