# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add unittest for functions in the analysis/spectrum_fitting/get_data.py file
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path
import unittest

# installed libraries
import pandas as pd

# local files for tests
import context
from analysis.spectrum_fitting import get_data


class TestGetC12880Data(unittest.TestCase):
    def test_get_background_n_dark(self):
        """
        Test that 'reference' and 'dark' data types return the correct DataFrame structure.

        This test:
        - Checks if the returned data is a Pandas DataFrame.
        - Verifies that the DataFrame has the expected shape (60, 290).
        - Ensures that the "sample" column contains all numbers from 1 to 60.
        """
        for _data_type in ["reference", "dark"]:
            data = get_data.get_raw_c12880_data(data_type=_data_type)

            # Assert that the data is a DataFrame
            self.assertIsInstance(data, pd.DataFrame, "Data should be a pandas DataFrame")

            # Assert that the shape is exactly (60, 290)
            self.assertEqual(data.shape, (60, 290),
                             f"Expected shape (60, 290), but got {data.shape}")

            # Expected set of values (1 to 60)
            expected_samples = set(range(1, 61))

            # Assert that unique samples match expected values
            self.assertEqual(
                set(data["sample"]), expected_samples,
                f"Missing values in 'sample' column: {expected_samples - set(data["sample"].values)}")

    def test_get_data(self):
        """
        Test that the 'data' type returns a properly formatted DataFrame.

        This test:
        - Checks if the returned data is a Pandas DataFrame.
        - Verifies that the DataFrame has the expected shape (960, 290).
        - Ensures that the "sample" column contains all numbers from 1 to 60.

        """
        data = get_data.get_raw_c12880_data(data_type="data")
        print(data)

        # Assert that the data is a DataFrame
        self.assertIsInstance(data, pd.DataFrame, "Data should be a pandas DataFrame")

        # Assert that the shape is exactly (60, 290)
        self.assertEqual(data.shape, (960, 290),
                         f"Expected shape (60, 290), but got {data.shape}")

        # Expected set of values (1 to 60)
        expected_samples = set(range(1, 61))

        # Assert that unique samples match expected values
        self.assertEqual(
            set(data["sample"]), expected_samples,
            f"Missing values in 'sample' column: {expected_samples - set(data["sample"].values)}")
