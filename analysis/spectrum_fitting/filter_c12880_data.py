# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Find the correct Savitzky-Golay (SG) filter parameters for the C12880 data, including background.
Use Fourier spectral smoothing
"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# local files
from get_data import get_raw_c12880_data


def foobar(data_type):
    data = get_raw_c12880_data(fruit="tomato", data_type=data_type)
    print(data.columns)
    x = data.drop(columns=["sample", "spot"])
    wavelengths = x.columns
    x_fourier = np.fft.fft(x, axis=1)

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    axs[0].plot(wavelengths, x[0, :])
    axs[1].semilogy(np.abs(x_fourier[0, :len(wavelengths)//2])**2)

    plt.show()


if __name__ == '__main__':
    foobar("reference")

