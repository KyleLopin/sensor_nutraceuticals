# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

# local files
from analysis.tests import context  # noqa - append sys.path
from analysis.spectrum_fitting import outlier_removal
from analysis.spectrum_fitting.outlier_removal import mahalanobis_outlier_removal


class TestCalculateSpectrumResidue(unittest.TestCase):
    """Test cases for the calculate_spectrum_residue function."""

    def test_simple(self):
        """Test the function with a simple dataset."""
        # Feature data
        x = pd.DataFrame([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]],
                         columns=["450 nm", "550 nm"])

        # Create a groups DataFrame with matching index
        groups = pd.DataFrame({
            "Fruit": [1, 1, 1],
            "spot": [1, 1, 1],
            "Read number": [1, 2, 3]
        }, index=x.index)

        # Expected correct residuals
        correct_df = pd.DataFrame([[-1.0, -1.0], [-1.0, -1.0], [2.0, 2.0]],
                                  columns=["450 nm", "550 nm"])
        results = outlier_removal.calculate_residues(x, groups)
        assert_frame_equal(results, correct_df, check_exact=False,
                           check_dtype=False)

    def test_2_groups(self):
        """Test the function with two separate groups."""
        # Feature data
        x = pd.DataFrame([
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 2.0]
        ], columns=["450 nm", "550 nm"])

        # Updated groups DataFrame
        groups = pd.DataFrame({
            "Fruit": [1, 1, 1, 2, 2, 2],
            "spot": [1, 1, 1, 1, 1, 1],
            "Read number": [1, 2, 3, 1, 2, 3]
        }, index=x.index)

        # Expected correct residuals
        correct_df = pd.DataFrame([
            [-1.0, -1.0],
            [-1.0, -1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [2.0, 2.0]
        ], columns=["450 nm", "550 nm"])

        results = outlier_removal.calculate_residues(x, groups)

        # Compare
        assert_frame_equal(results, correct_df, check_exact=False, check_dtype=False)


class TestCalculateResidues(unittest.TestCase):
    """
    More unit tests for the calculate_residues function, which computes residuals
    relative to group-wise means. Uses setUp so can not combine easily with other
    TestCase.
    """

    def setUp(self):
        """
        Prepare small example datasets for testing.
        """
        # Small example x: 4 rows × 2 features
        self.x = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0]
        })

        # Matching groups: (Fruit, spot, read_number)
        self.groups = pd.DataFrame({
            "Fruit": [1, 1, 2, 2],
            "spot": [1, 1, 2, 2],
            "Read number": [1, 2, 1, 2]
        })

        self.groups.index = self.x.index  # match index

    def test_residues_shape_and_index(self):
        """
        Test that the returned DataFrame has the same shape, index, and columns as input.
        """
        residues = outlier_removal.calculate_residues(self.x, self.groups)

        # Check shape and structure
        self.assertEqual(residues.shape, self.x.shape, "Shape mismatch")
        self.assertTrue(residues.index.equals(self.x.index), "Index mismatch")
        self.assertListEqual(list(residues.columns), list(self.x.columns), "Column names mismatch")

    def test_residue_values(self):
        """
        Test that the calculated residues are correct for a simple case.
        """
        residues = outlier_removal.calculate_residues(self.x, self.groups)

        # Manually calculate expected values
        # Group 1, spot 1: rows 0 and 1
        expected_row0 = self.x.loc[0] - self.x.loc[[1]].mean()
        expected_row1 = self.x.loc[1] - self.x.loc[[0]].mean()
        # Group 2, spot 2: rows 2 and 3
        expected_row2 = self.x.loc[2] - self.x.loc[[3]].mean()
        expected_row3 = self.x.loc[3] - self.x.loc[[2]].mean()

        # Check values
        pd.testing.assert_series_equal(residues.loc[0], expected_row0, check_names=False)
        pd.testing.assert_series_equal(residues.loc[1], expected_row1, check_names=False)
        pd.testing.assert_series_equal(residues.loc[2], expected_row2, check_names=False)
        pd.testing.assert_series_equal(residues.loc[3], expected_row3, check_names=False)

    def test_single_row_group(self):
        """
        Test behavior when there is only one row in a group (should not fail).
        """
        # Modify groups: make one unique
        modified_groups = self.groups.copy()
        modified_groups.loc[0, "Fruit"] = 99  # Unique fruit for row 0

        residues = outlier_removal.calculate_residues(self.x, modified_groups)

        # The residue for row 0 should be NaN (no other rows to compare)
        self.assertTrue(residues.loc[0].isnull().all(),
                        "Residue should be NaN for singleton group")

    def test_input_series(self):
        """
        Test behavior if x is a Series instead of a DataFrame.
        """
        x_series = self.x["feature1"]  # Just one column as Series

        residues = outlier_removal.calculate_residues(x_series, self.groups)

        self.assertIsInstance(residues, pd.Series,
                              "Output should be Series if input is Series")
        self.assertEqual(residues.shape, x_series.shape, "Shape mismatch on Series input")


class TestMahalanobisOutlierRemoval(unittest.TestCase):
    """Unit tests for the mahalanobis_outlier_removal function."""

    def setUp(self):
        """Create clean test data with known outliers."""
        # Normal cluster of points (centered at (0, 0))
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, size=(50, 2))  # 50 points

        # Add obvious outliers far from cluster
        outliers = np.array([
            [10, 10],
            [-10, -10],
            [15, -15]
        ])

        # Combine into one dataset
        self.x = pd.DataFrame(
            np.vstack([normal_data, outliers]),
            columns=["feature1", "feature2"]
        )

    def test_detect_outliers(self):
        """Test that obvious outliers are detected."""
        mask = mahalanobis_outlier_removal(self.x, use_robust=True,
                                           display_hist=False, cutoff_limit=3.0)

        # There are 53 points total
        self.assertEqual(mask.shape[0], self.x.shape[0], "Mask length should match input rows")

        # Expect at least the 3 outliers to be flagged as False (masked out)
        num_outliers_detected = (~mask).sum()
        print("===", num_outliers_detected)
        self.assertGreaterEqual(num_outliers_detected, 3, "Should detect at least 3 outliers")

    def test_no_outliers_in_clean_data(self):
        """Test that clean normal data has very few false positives."""
        np.random.seed(0)
        clean_data = pd.DataFrame(
            np.random.normal(0, 1, size=(100, 2)),
            columns=["feature1", "feature2"]
        )
        mask = mahalanobis_outlier_removal(clean_data, use_robust=True, display_hist=False, cutoff_limit=4.0)

        # In purely normal data, few or no outliers should be found
        num_outliers = (~mask).sum()
        self.assertLessEqual(num_outliers, 5, "Should not detect many false outliers in clean data")




