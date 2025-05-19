# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Check correlation between target variables to check how they vary together.
Hypothesis: beta-carotene and lycopene are highly correlated as they are
both a function of plant health
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path
import sys

# installed libraries
import pandas as pd

# local files
sys.path.append(str(Path(__file__).resolve().parent.parent))
from spectrum_fitting import get_data

def get_lyco_care_r2():
    sensor, fruit = "as7262", "tomato"
    targets = pd.DataFrame()
    for target in ["lycopene (FW)", "beta-carotene (FW)"]:
        x, y, groups = get_data.get_data(sensor=sensor,
                                         fruit=fruit,
                                         # measurement_mode=measurement_type,
                                         target_column=target,
                                         mean_spot=True)
        targets[target] = y
    print(targets)
    print("r2 score", targets.corr())


if __name__ == '__main__':
    get_lyco_care_r2()
