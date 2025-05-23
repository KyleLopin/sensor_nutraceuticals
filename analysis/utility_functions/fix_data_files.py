# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import pandas as pd
DATA_FOLDER = Path(__file__).parent.parent.parent / "data"
print(DATA_FOLDER)


def fix_mango_files():
    data_folder = DATA_FOLDER / "mango" / "reflectance_orig"
    for sensor in ["as7262", "as7263", "as7265x"]:
        data = pd.read_csv(data_folder / f"mango_{sensor}_ref_data.csv")
        print(data.columns)
        data = data.drop(columns=["Unnamed: 0"])
        print(data[["Fruit", "Fruit number"]])
        data["spot"] = data["Fruit number"].str.split(' ').str[3]
        print(data.columns)



if __name__ == '__main__':
    fix_mango_files()
