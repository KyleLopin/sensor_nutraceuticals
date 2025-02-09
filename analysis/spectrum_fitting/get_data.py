# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Load spectral data sets for tomato and mango fruit.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import pandas as pd

DATA_FOLDER = Path(__file__).parent.parent.parent / "data"
pd.set_option('display.max_rows', 10)


# def get_data(sensor: str, fruit: str):
#     filename = DATA_FOLDER / f"{fruit}_{sensor}_data.csv"


def get_raw_c12880_data(fruit: str = "tomato", data_type: str = "data") -> pd.DataFrame:
    """
    Load raw spectrometer from C12880 on tomatoes and mangos.

    This function reads an dataset for a given fruit and filters the data based
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

    Returns
    -------
    pd.DataFrame
        A processed DataFrame containing the relevant spectral data, with:
        - `"sample"` : Sample numbers
        - `"spot"` : Spot number ont he sample.
        - Remaining columns : Spectral intensity values at different wavelengths.

    Raises
    ------
    ValueError
        If `data_type` is not one of `"reference"`, `"dark"`, or `"data"`.

    Notes
    -----
    - The `sample` column is forward-filled because it was not fully entered during data collection.

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
    # the excel file has 2 top rows merged, remove the first one
    data = pd.read_excel(data_path, skiprows=1)
    # then add in the names that were removed with skiprows=1
    data.columns = ["sample", "spot"] + list(data.columns[2:])
    # fill in sample numbers down the column, they were not entered during data collection
    # and fix that read_excels converts ints to floats
    data["sample"] = data["sample"].fillna(method="ffill").astype(int)
    if data_type == "reference":
        data = data[data["spot"] == "White"]
    elif data_type == "dark":
        data = data[data["spot"] == "dark"]
    elif data_type == "data":
        data = data[data["spot"].astype(str).str.contains(r"^\d+\.\d+$", na=False)]
    else:
        raise ValueError(f"data_type = {data_type} is not valid, data_type must be in "
                         f"['reference', 'dark', 'data']")

    return data


if __name__ == '__main__':
    get_raw_c12880_data(data_type="dark")
