# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Load spectral data sets for tomato and mango fruit.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import numpy as np
import pandas as pd

DATA_FOLDER = Path(__file__).parent.parent.parent / "data"
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', None)


# def get_data(sensor: str, fruit: str):
#     filename = DATA_FOLDER / f"{fruit}_{sensor}_data.csv"


def is_float(value):
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
    data = pd.read_excel(data_path, skiprows=1)

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

    # because the sensor has to be programmed with wavelenghts, some numbers are repeated
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
        # get the dark current, take mean of every wavelenght
        # less than the dark current cutoff
        dark_current_levels = data[
            [wl for wl in spectral_columns
             if wl < dark_current_cutoff]
        ]
        dark_current_cols = [wl for wl in spectral_columns
                             if wl < dark_current_cutoff]
        # print(dark_current_cols)

        # Compute a **single** dark current value per sample (mean across all dark current columns)
        dark_current_levels = data[dark_current_cols].mean(axis=1)
        # print(dark_current_levels)

        # Subtract the sample-specific dark current from the corresponding data rows
        # data = data.merge(dark_current_levels, on="sample")  # Merge to align by sample
        # print(data)
        data[spectral_columns] = data[spectral_columns].sub(dark_current_levels, axis=0)
        # data = data.drop(columns=["dark_current"])

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
                     use_individual_refl: bool = False,
                     mean_spot: bool = False,
                     target: str = "lycopene (DW)",
                     wavelength_range: tuple[float, float] = (412, 691),
                     **kwargs):
    print("getting c12880")
    if measurement_mode == "raw":
        data = get_raw_c12880_data(fruit=fruit,
                                   data_type="data",
                                   **kwargs)
        # TODO: get this part working if needed
    elif measurement_mode == "reflectance":
        data_folder = DATA_FOLDER / fruit / "reflectance"
        data_file = data_folder / "tomato_reflectance_single.csv"
        if use_individual_refl:
            data_file = data_folder / "tomato_reflectance_individual.csv"
        data = pd.read_csv(data_file)
        print(data)
        print(data.columns)
        # get Y from the other datasets
        print('kwarg: ', kwargs)

    data["spot_group"] = data["spot"].astype(str).str.split(".").str[0].astype(int)
    agg_dict = {col: 'mean' for col in data.columns}
    agg_dict["sample"] = "first"
    agg_dict["spot"] = "first"
    if mean_spot:
        data = data.groupby(by=["sample", "spot_group"]).mean()
        print("mean data")
        print(data)
    invalid_columns = [col for col in data.columns
                       if not is_float(col)]
    x = data.drop(columns=invalid_columns)
    x.columns = [float(col) for col in x.columns]
    if wavelength_range:
        filtered_columns = [
            wl for wl in x.columns
            if isinstance(wl, (int, float)) and wavelength_range[0] < wl < wavelength_range[1]
        ]
        print("fc: ", filtered_columns, x.columns)
        x = x[filtered_columns]
    print(x)
    # get y from as7262 data
    y_data_df = pd.read_csv(data_folder / "tomato_as7262_ref_data.csv")
    y = y_data_df.groupby(by=["Fruit"]).agg({target: "first"})
    y.index.name = "sample"
    print(y)
    groups = data.index.to_frame(index=False)
    print(groups)
    df_merge =groups.merge(y.reset_index(), on="sample", how="left")
    y = df_merge[target]
    print(df_merge)
    print(x.shape, y.shape, groups.shape, df_merge.shape)
    return x, y, groups


def get_data(sensor: str,
             measurement_mode: str = "reflectance",
             fruit: str = "tomato",
             target_column="lycopene (DW)",
             split_x_y_groups: bool = True,
             mean_spot: bool = False,
             **kwargs):
    if sensor == "c12880":
        print('bb')
        return _get_c12880_data(fruit=fruit,
                                measurement_mode=measurement_mode,
                                mean_spot=mean_spot,
                                target=target_column,
                                **kwargs)
        # TODO; GET Y
        groups = data[["sample", "spot"]]
        x = data.drop(columns=["sample", "spot"])

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
    print(data)

    # get spectral columns
    x_columns = []
    for column in data.columns:
        if 'nm' in column:
            x_columns.append(column)

    if target_column not in data.columns:
        raise ValueError(f"{target_column} is not in DataFrame, valid columns are: {data.columns}")

    if kwargs:
        print("got sensor settings:", kwargs)
        for sensor_setting, sensor_value in kwargs.items():
            data = data.loc[data[sensor_setting] == sensor_value]

    if mean_spot:
        agg_dict = {col: "mean" for col in x_columns}
        for new_col in ["Fruit", "Fruit number", "spot", "Read number", target_column]:
            agg_dict[new_col] = "first"
        data = data.groupby(by=["Fruit number"]).agg(agg_dict)
        print("mean data")
        print(data)

    x = data[x_columns]
    if measurement_mode == "absorbance":
        x = -np.log10(x)
    print(data.columns)

    print(data)
    if split_x_y_groups:
        return x, data[target_column], data[["Fruit", "spot", "Read number"]]
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sensor_settings = {"led current": "12.5 mA",
                       "integration time": 50}
    x, y, groups = get_data("as7262", measurement_mode="raw",
                            mean_spot=True,
                            **sensor_settings)

    # print(x)
    # print(y)
    # print(groups)
    plt.plot(x.T)
    plt.show()

    data = get_raw_c12880_data(data_type="data",
                               dark_current_cutoff=360,
                               wavelength_range=None)
    # print(data.columns)
    # print(data)
    # data = get_raw_c12880_data(data_type="data", wavelength_range=None)
    plt.plot(data.drop(columns=["spot", "sample"]).T)
    plt.show()
