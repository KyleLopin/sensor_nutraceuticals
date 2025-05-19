# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add the proper directory to python path for tests to work
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import pathlib
import sys

parent1 = pathlib.Path().absolute().parent
spectral1 = parent1 / "spectrum_fitting"
if spectral1.exists():
    sys.path.append(str(spectral1))
else:
    spectral2 = parent1.parent / "spectrum_fitting"
    if spectral2.exists():
        sys.path.append(str(spectral2))
    else:
        raise FileNotFoundError(
            f"Could not find 'spectrum_fitting' directory in {parent1} or {parent1.parent}"
        )
# spectrum_file_folder = pathlib.Path().absolute().parent / "spectrum_fitting"
print(sys.path)