# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add the proper directory to python path for tests to work
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import pathlib
import sys

spectrum_file_folder = pathlib.Path().absolute().parent / "spectrum_fitting"
sys.path.append(str(spectrum_file_folder))
