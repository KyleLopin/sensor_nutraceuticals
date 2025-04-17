# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add unittest for functions in the analysis/spectrum_fitting/get_data.py file
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files for tests
import context  # noqa
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


class TestGetDataIndexConsistency(unittest.TestCase):
    def setUp(self):
        self.sensor = "as7262"
        # self.sensor = "c12880"
        self.fruit = "tomato"
        self.measurement_mode = "reflectance"
        self.target = "lycopene (FW)"

    def _load_data(self, mean_spot: bool):
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

    def test_groups_columns(self):
        for sensor in ["as7262", "as7263", "as7265x", "c12880"]:
            with self.subTest(sensor=sensor):
                self.sensor = sensor  # override
                self._test_group_columns()

    def _test_group_columns(self):
        _, _, groups = self._load_data(mean_spot=True)
        # print(groups.columns)
        expected_columns = ['Fruit', 'spot', 'Read number']
        self.assertListEqual(list(groups.columns), expected_columns,
                             "groups columns do not match expected order "
                             "and names for meaned data")
        _, _, groups = self._load_data(mean_spot=False)
        # print(groups.columns)
        expected_columns = ['Fruit', 'spot', 'Read number']
        self.assertListEqual(list(groups.columns), expected_columns,
                             "groups columns do not match expected order "
                             "and names for data not averaged")

    # def test_mean_spot_true_index_alignment(self):
    #     # for sensor in ["as7262", "as7263", "as7265x", "c12880"]:
    #     for sensor in ["as7262", "c12880"]:
    #         with self.subTest(sensor=sensor):
    #             self.sensor = sensor  # override
    #             x, y, groups = self._load_data(mean_spot=True)
    #
    #             # Check types
    #             self.assertIsInstance(x, pd.DataFrame)
    #             self.assertIsInstance(y, pd.Series)
    #             self.assertIsInstance(groups, pd.DataFrame)
    #
    #             # Check that all have the same index
    #             print("indexes")
    #             print(x.index[:5])
    #             print(y.index[:5])
    #             print(groups.index[:5])
    #             # print(x.head())
    #             # print(y.head())
    #             print(groups.head())
    #             self.assertTrue(x.index.equals(y.index), "x and y index mismatch")
    #             self.assertTrue(x.index.equals(groups.index), "x and groups index mismatch")
    #
    #             # Check that index is MultiIndex with correct names
    #             self.assertNotIsInstance(x.index, pd.MultiIndex)
    #             # self.assertEqual(x.index.names, ['sample', 'spot_group'], "Unexpected index names")
    #
    # def test_mean_spot_false_index_alignment(self):
    #     x, y, groups = get_data.get_data(
    #         sensor=self.sensor,
    #         fruit=self.fruit,
    #         measurement_mode=self.measurement_mode,
    #         target_column=self.target,
    #         mean_spot=False
    #     )
    #
    #     self.assertIsInstance(x, pd.DataFrame)
    #     self.assertIsInstance(y, pd.Series)
    #     self.assertIsInstance(groups, pd.DataFrame)
    #
    #     # Check index consistency
    #     self.assertTrue(x.index.equals(y.index), "x and y index mismatch")
    #     self.assertTrue(x.index.equals(groups.index), "x and groups index mismatch")
    #
    #     # Check for expected index names or columns
    #     print(x.index.names)
    #     # self.assertIn("Fruit", x.index.names)
    #     # self.assertIn("spot", x.index.names)
    #
    # def test_index_structure_and_equality(self):
    #     x, y, groups = self._load_data(mean_spot=True)  # or False
    #     print("print statements")
    #     print(x.index)
    #     print(y.index)
    #     print(groups.index)
    #     # self.assertEqual(1, 2, "Force fail to view output")
    #
    #     # Check that all indexes are equal to x.index
    #     for name, obj in zip(["y", "groups"], [y, groups]):
    #         self.assertTrue(
    #             x.index.equals(obj.index),
    #             f"x and {name} index mismatch"
    #         )
    #
    #     # Check each has a single-level index with name "Fruit number"
    #     for name, obj in zip(["x", "y", "groups"], [x, y, groups]):
    #         self.assertFalse(
    #             isinstance(obj.index, pd.MultiIndex),
    #             f"{name} index should not be a MultiIndex"
    #         )
    #         self.assertEqual(
    #             obj.index.nlevels, 1,
    #             f"{name} index should have exactly 1 level"
    #         )
    #         self.assertEqual(
    #             obj.index.name, "Fruit number",
    #             f"{name} index should be named 'Fruit number'"
    #         )

