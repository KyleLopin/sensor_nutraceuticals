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


def vis_data_and_fourier_transform(data_type: str, spectra_indices: list[int] = [0]) -> None:
    """
    Visualize raw spectral data, and it Fourier Transform.

    Based on: https://nirpyresearch.com/optimal-spectra-smoothing-fourier-ring-correlation/

    Parameters:
    -----------
    data_type : str
        Type of spectral data to fetch ("reference", "data", "dark").
    spectra_indices : list[int]
        Indices of data to show

    Steps:
    ------
    - Retrieves spectral data for a given fruit (tomato).
    - Drops unwanted columns ('sample' and 'spot').
    - Cleans wavelength labels, converting them to floats if needed.
    - Computes the Fast Fourier Transform (FFT) along the wavelength axis.
    - Plots the original spectrum and its FFT power spectrum.

    Returns:
    --------
    None (displays plots).
    """
    data = get_raw_c12880_data(fruit="tomato", data_type=data_type)
    print(data.columns)
    x = data.drop(columns=["sample", "spot"])
    # wavelengths = x.columns
    # Clean the wavelength column names
    wavelengths = [float(wl[:-2]) if isinstance(wl, str) else wl for wl in x.columns]
    x_fourier = np.fft.fft(x, axis=1)

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    x = x.values

    for i in spectra_indices:
        axs[0].plot(wavelengths, x[i, :])
        axs[1].semilogy(np.abs(x_fourier[i, :len(wavelengths)//2])**2)
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Counts")

    axs[1].set_xlabel("Fourier coordinate")

    plt.show()


def plot_spectrum_histogram(data_type: str, threshold: float = None) -> None:
    """
    Plot a histogram of spectral intensity values from the specified data type
    and print the mean and standard deviation.

    Parameters:
    -----------
    data_type : str
        Type of spectral data to fetch ("reference" or "dark").
    threshold : Optional[float], default=None
        If provided and data_type is "reference", a red vertical line is added to the histogram.

    Steps:
    ------
    - Retrieves spectral data for a given fruit (tomato).
    - Drops unwanted columns ('sample' and 'spot').
    - Flattens all intensity values into a 1D array.
    - Computes and prints the mean and standard deviation (if "dark").
    - Plots a histogram of intensity values.
    - If "reference" data type and threshold is provided, a vertical red line is drawn.

    Returns:
    --------
    None (prints statistics and displays a histogram).
    """
    if data_type not in ["reference", "dark"]:
        raise ValueError("data_type must be either 'reference' or 'dark'.")

    # Retrieve spectral data
    data = get_raw_c12880_data(fruit="tomato", data_type=data_type)

    # Drop non-spectral columns
    x = data.drop(columns=["sample", "spot"])

    # Flatten the intensity values into a 1D array
    intensity_values = x.values.flatten()

    if data_type == "dark":
        # Compute statistics
        mean_intensity = np.mean(intensity_values)
        std_intensity = np.std(intensity_values)

        # Print statistics
        print(f"Mean Intensity ({data_type}): {mean_intensity:.2f}")
        print(f"Standard Deviation ({data_type}): {std_intensity:.2f}")

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(intensity_values, bins=100, alpha=0.75, color='blue', edgecolor='black')

    # Add a vertical threshold line if data_type is "reference"
    if data_type == "reference" and threshold is not None:
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f"Threshold = {threshold}")
        plt.legend()

    # Labels and title
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {data_type.capitalize()} Spectral Intensity")

    plt.grid(True)
    plt.show()


def plot_spectrum_with_threshold(data_type: str, wavelength: float) -> None:
    """
    Plot spectral data, analyze values below a given wavelength, and highlight extreme values.

    Parameters:
    -----------
    data_type : str
        Type of spectral data to fetch ("reference" or "dark").
    wavelength : float or int
        Wavelength at which to draw a vertical line.
        The mean and standard deviation are computed for all wavelengths < this value.
        Additionally, wavelengths exceeding mean + 5 * std are highlighted.

    Steps:
    ------
    - Retrieves spectral data for a given fruit (tomato).
    - Drops unwanted columns ('sample' and 'spot').
    - Extracts wavelengths from column names.
    - Computes mean and standard deviation for all wavelengths < `wavelength`.
    - Plots the spectral data with a vertical line at `wavelength`.
    - Adds a horizontal line at `mean + 5 * std`.
    - Identifies wavelengths where spectral intensity exceeds `mean + 5 * std`.

    Returns:
    --------
    None (prints statistics and displays a plot).
    """
    if data_type not in ["reference", "dark"]:
        raise ValueError("data_type must be either 'reference' or 'dark'.")

    # Retrieve spectral data
    data = get_raw_c12880_data(fruit="tomato", data_type=data_type,
                               wavelength_range=None)

    # Drop non-spectral columns
    x = data.drop(columns=["sample", "spot"])

    # Extract and clean wavelength column names
    wavelengths = [float(wl[:-2]) if isinstance(wl, str) and wl.endswith('.1') else float(wl) for wl in x.columns]

    # Convert data to NumPy array
    x_values = x.values

    # Find indices of wavelengths below the given threshold
    valid_indices = [i for i, wl in enumerate(wavelengths) if wl < wavelength]

    if not valid_indices:
        print(f"No wavelengths found below {wavelength} nm.")
        return

    # Compute statistics for those wavelengths
    selected_data = x_values[:, valid_indices]  # Select columns with wavelengths < threshold
    mean_intensity = np.mean(selected_data)
    std_intensity = np.std(selected_data)
    threshold_intensity = mean_intensity + 20 * std_intensity

    # Print statistics
    print(f"Mean Intensity for λ < {wavelength} nm: {mean_intensity:.2f}")
    print(f"Standard Deviation for λ < {wavelength} nm: {std_intensity:.2f}")
    print(f"Threshold (Mean + 20d*Std): {threshold_intensity:.2f}")

    # Find wavelengths where intensity exceeds threshold
    high_intensity_wavelengths = []
    for i, wl in enumerate(wavelengths):
        if np.any(x_values[:, i] > threshold_intensity):
            high_intensity_wavelengths.append(wl)

    # Split into left and right based on the vertical threshold wavelength
    # left_side = [wl for wl in high_intensity_wavelengths if wl < wavelength]
    # right_side = [wl for wl in high_intensity_wavelengths if wl > wavelength]

    print(f"Wavelengths exceeding threshold on LEFT side (λ < {wavelength}): {high_intensity_wavelengths[0]}")
    print(f"Wavelengths exceeding threshold on RIGHT side (λ > {wavelength}): {high_intensity_wavelengths[-1]}")

    # Plot spectral data
    plt.figure(figsize=(8, 5))
    for spectrum in x_values:
        plt.plot(wavelengths, spectrum, alpha=0.3, color="blue")  # Plot all spectra lightly

    # Vertical threshold line
    plt.axvline(wavelength, color='red', linestyle='dashed', linewidth=2,
                label=f"Threshold λ = {wavelength} nm")

    # Horizontal threshold line
    plt.axhline(threshold_intensity, color='green', linestyle='dotted', linewidth=2,
                label=f"Threshold = {threshold_intensity:.2f}")
    plt.axvline(412, color='blue', linestyle='dotted', linewidth=2)

    plt.axvline(691, color='blue', linestyle='dotted', linewidth=2)

    # Labels and title
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum Data with Threshold at {wavelength} nm")
    plt.legend()
    plt.grid(True)
    # plt.xlim([390, 420])
    print("what?")
    plt.ylim([500, 750])
    plt.show()


if __name__ == '__main__':
    vis_data_and_fourier_transform("reference", range(50))
    # plot_spectrum_histogram("reference", threshold=425+5*8.5)
    # plot_spectrum_with_threshold("reference", 380)
