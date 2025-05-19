# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add unittest for functions in the analysis/spectrum_fitting/get_data.py file
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from itertools import product
import unittest

# installed libraries
import pandas as pd

# local files for tests
from analysis.tests import context  # noqa
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
        test_cases = [[None, (60, 288)], ["default", (60, 129)]]
        for test_case in test_cases:
            for _data_type in ["reference", "dark"]:
                if test_case[0] == "default":
                    data = get_data.get_raw_c12880_data(data_type=_data_type)
                else:
                    data = get_data.get_raw_c12880_data(data_type=_data_type,
                                                        wavelength_range=test_case[0])

                # Assert that the data is a DataFrame
                self.assertIsInstance(data, pd.DataFrame, "Data should be a pandas DataFrame")

                # Assert that the shape is exactly (60, 290)
                self.assertEqual(data.shape, test_case[1],
                                 f"Expected shape {test_case[1]}, but got {data.shape}"
                                 f"for the case of {test_case[0]}")

                # Expected set of values (1 to 60)
                expected_samples = set(range(1, 61))

                # Assert that unique samples match expected values
                self.assertEqual(
                    set(data["sample"]), expected_samples,
                    f"Missing values in 'sample' column: "
                    f"{expected_samples - set(data['sample'].values)}")

    def test_get_data(self):
        """
        Test that the 'data' type returns a properly formatted DataFrame.

        This test:
        - Checks if the returned data is a Pandas DataFrame.
        - Verifies that the DataFrame has the expected shape (960, 290).
        - Ensures that the "sample" column contains all numbers from 1 to 60.

        """
        test_cases = [[None, (60, 288)], ["default", (60, 129)]]
        for test_case in test_cases:
            for _data_type in ["reference", "dark"]:
                if test_case[0] == "default":  # test not passing wavelength_range
                    data = get_data.get_raw_c12880_data(data_type=_data_type)
                else:
                    data = get_data.get_raw_c12880_data(data_type=_data_type,
                                                        wavelength_range=test_case[0])

            # Assert that the data is a DataFrame
            self.assertIsInstance(data, pd.DataFrame, "Data should be a pandas DataFrame")

            # Assert that the shape is exactly (60, 290)
            self.assertEqual(data.shape, test_case[1],
                             f"Expected shape {test_case[1]}, but got {data.shape}"
                             f"for the condition: {test_case[0]}")

            # Expected set of values (1 to 60)
            expected_samples = set(range(1, 61))

            # Assert that unique samples match expected values
            self.assertEqual(
                set(data["sample"]), expected_samples,
                f"Missing values in 'sample' column: "
                f"{expected_samples - set(data['sample'].values)}")


class TestGetC12880DataFromGetData(unittest.TestCase):
    def test_get_x_y(self):
        """Loop over combinations of fruit, mean_spot, and measurement_mode."""
        fruits = ["mango", "tomato"]
        for fruit in fruits:
            with self.subTest(fruit=fruit):
                self._check_all_cases(fruit)

    def _check_all_cases(self, fruit: str):
        """
        Helper function to load data and assert structure and consistency.

        Parameters
        ----------
        fruit : str
            Fruit name ("mango", "tomato").
        mean_spot : bool
            Whether to average spot measurements.
        method : str
            Measurement mode ("raw" or "reflectance").
        """
        target = "lycopene (DW)"
        if fruit == "mango":
            target = "carotene (DW)"
        x, y, groups = get_data.get_data(
            sensor="c12880", fruit=fruit, target_column=target,
            measurement_mode="reflectance", mean_spot=True)
        print(x.columns)
        print([x for x in x.columns if not isinstance(x, float)])
        is_float = (isinstance(x, float) for x in x.columns)
        self.assertTrue(all(is_float), msg=f"is_float = {list(is_float)}\nfor columns:{x.columns}")
        # print(y)
        # print(groups)


class TestGetDataIndexConsistency(unittest.TestCase):
    """
    Unit tests for validating the consistency of index alignment and column structure
    returned by the `get_data.get_data` function across different sensors and
    spot averaging configurations.
    """

    def setUp(self):
        """
        Set up default sensor parameters used across test cases.
        Initializes the sensor type, fruit, measurement mode, and target.
        """
        self.sensor = "as7262"  # just a holder, the methods will vary
        self.fruit = "tomato"
        self.measurement_mode = "reflectance"
        self.target = "lycopene (FW)"

    def _load_data(self, mean_spot: bool):
        """
        Internal helper method to load data using the get_data interface with
        standardized sensor settings and the option to control spot averaging.

        Parameters
        ----------
        mean_spot : bool
            Whether to average spot readings before returning data.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.DataFrame]
            Feature data (x), target values (y), and metadata (groups).
        """
        sensor_settings = {"led current": "12.5 mA",
                           "integration time": 50}
        if self.sensor == "as7265x":
            sensor_settings["led"] = "b'White IR'"
        return get_data.get_data(
            sensor=self.sensor,
            fruit=self.fruit,
            measurement_mode=self.measurement_mode,
            target_column=self.target,
            mean_spot=mean_spot,
            **sensor_settings
        )

    def test_group_columns(self):
        """
        Test that the 'groups' DataFrame returned by get_data has the correct columns
        depending on the `mean_spot` setting. When spot readings are averaged, the
        group should only contain ['Fruit', 'spot'], otherwise it should include
        ['Fruit', 'spot', 'Read number'].
        """
        for sensor in ["as7262", "as7263", "as7265x", "c12880"]:
            with self.subTest(sensor=sensor):
                self.sensor = sensor  # override
                _, _, groups = self._load_data(mean_spot=True)
                # print(groups.columns)
                expected_columns = ['Fruit', 'spot']
                self.assertListEqual(list(groups.columns), expected_columns,
                                     "groups columns do not match expected order "
                                     "and names for averaged data")
                _, _, groups = self._load_data(mean_spot=False)
                # print(groups.columns)
                expected_columns = ['Fruit', 'spot', 'Read number']
                self.assertListEqual(list(groups.columns), expected_columns,
                                     "groups columns do not match expected order "
                                     "and names for data not averaged")

    def test_mean_spot_true_index_alignment(self):
        """
        Test that the feature (x), target (y), and metadata (groups) DataFrames
        all have the same flat index (not a MultiIndex) and are aligned correctly
        when using both `mean_spot=True` and `mean_spot=False`.
        """
        for sensor in ["as7262", "as7263", "as7265x", "c12880"]:
        # for sensor in ["as7262", "c12880"]:
            for mean_spot in [True, False]:
                with self.subTest(sensor=sensor):
                    self.sensor = sensor  # override
                    x, y, groups = self._load_data(mean_spot=mean_spot)

                    # Check types
                    self.assertIsInstance(x, pd.DataFrame)
                    self.assertIsInstance(y, pd.DataFrame,
                                          msg=f"\ny should be pandas DataFrame, it is type {type(y)}")
                    self.assertIsInstance(groups, pd.DataFrame)

                    # Check that all have the same index
                    self.assertTrue(x.index.equals(y.index), "x and y index mismatch\n"
                                                             f"x.index: {x.index}\n"
                                                             f"y.index: {y.index}")
                    self.assertTrue(x.index.equals(groups.index), "x and groups index mismatch")

                    # Check that index is not MultiIndex (the groupby has to be reset)
                    self.assertNotIsInstance(x.index, pd.MultiIndex)


class TestBasicLoad(unittest.TestCase):
    def test_basic_load(self):
        """Loop over combinations of fruit, mean_spot, and measurement_mode."""
        fruits = ["mango", "tomato"]
        sensors = ["as7262", "as7263", "as7265x", "c12880"]

        for sensor, fruit, in product(sensors, fruits):
            with self.subTest(sensor=sensor, fruit=fruit):
                target = "lycopene (DW)"
                if fruit == "mango":
                    target = "carotene (DW)"
                _, _, _ = get_data.get_data(
                    sensor=sensor, fruit=fruit,
                    target_column=target,
                    measurement_mode="reflectance", mean_spot=True)
                self.assertTrue(True)
