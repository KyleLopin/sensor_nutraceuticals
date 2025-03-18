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

FRUIT_COLOR = "red"


def fourier_ring_correlation(fft1: np.ndarray, fft2: np.ndarray) -> np.ndarray:
    """
    Compute Fourier Ring Correlation (FRC) between two 1D spectra across frequency bins.

    FRC measures the similarity between two spectra in the frequency domain.
    It is commonly used to assess signal smoothing effectiveness.

    Parameters:
    -----------
    fft1 : np.ndarray
        FFT of the first spectrum (original signal).
    fft2 : np.ndarray
        FFT of the second spectrum (filtered signal).

    Returns:
    --------
    np.ndarray
        A 1D array of FRC values for each frequency bin.

    Notes:
    ------
    - The FRC values range from 0 (no correlation) to 1 (perfect correlation).
    - Uses only the first half of the FFT spectrum (positive frequencies).
    - The first frequency bin (DC component) is excluded to avoid bias.

    Reference:
    ----------
    - https://nirpyresearch.com/optimal-spectra-smoothing-fourier-ring-correlation/
    """
    frc = np.zeros(fft1.shape[0] // 2 - 1).astype('complex128')
    for i in range(1, fft1.shape[0] // 2, 1):
        frc_num = fft1[i] * np.conjugate(fft2[i])
        norm1 = np.abs(fft1[i]) ** 2
        norm2 = np.abs(fft2[i]) ** 2
        frc[i - 1] = frc_num / np.sqrt(norm1 * norm2)
    print(frc.shape)
    return np.real(frc)


def vis_data_and_fourier_transform(data_type: str,
                                   spectra_indices: list[int] = [0],
                                   sg_filter_params: dict[str: int] = None) -> None:
    """
    Visualize raw spectral data, its Fourier Transform, and Fourier Ring Correlation (FRC).

    Based on: https://nirpyresearch.com/optimal-spectra-smoothing-fourier-ring-correlation/

    Parameters:
    -----------
    data_type : str
        Type of spectral data to fetch ("reference", "data", "dark").
    spectra_indices : list[int]
        Indices of data to show
    sg_filter_params : dict[str, int] | None
        If provided, applies a Savitzky-Golay filter to smooth the signal.
        Expected keys:
        - "window_length": int (must be odd)
        - "polyorder": int (polynomial order)

    Steps:
    ------
    - Retrieves spectral data for a given fruit (tomato).
    - Drops unwanted columns ('sample' and 'spot').
    - Cleans wavelength labels, converting them to floats if needed.
    - Computes the Fast Fourier Transform (FFT) along the wavelength axis.
    - Plots the original spectrum and its FFT power spectrum.
    - If `sg_filter_params` is provided, overlays the smoothed signal and its FFT.
    - Computes and plots Fourier Ring Correlation (FRC) between original and filtered spectra.


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
    # x_fourier = np.fft.fft(x, axis=1)

    fig, axs = plt.subplots(3, 1, figsize=(6, 9))
    x = x.values
    frc_values = []
    for i in spectra_indices:
        original_spectrum= x[i, :]
        # Compute FFT of original spectrum
        original_fft = np.fft.fft(original_spectrum)

        axs[0].plot(wavelengths, original_spectrum, color='black')
        axs[0].set_title("Original Spectrum")
        axs[1].semilogy(np.abs(original_fft[:len(wavelengths)//2])**2, color='black',
                        label="Original data")
        axs[1].set_title("Fourier Transform of Spectral Data")
        if sg_filter_params:
            filtered_spectrum = savgol_filter(original_spectrum, **sg_filter_params)
            # Compute FFT of filtered spectrum
            filtered_fft = np.fft.fft(filtered_spectrum)

            # Overlay filtered signal on original plot
            axs[0].plot(wavelengths, filtered_spectrum, linestyle="dashed", c=FRUIT_COLOR)
            # Plot filtered FFT
            axs[1].semilogy(np.abs(filtered_fft[:len(wavelengths) // 2]) ** 2,
                            linestyle="dashed", color=FRUIT_COLOR, label="Filtered data")
            # Compute Fourier Ring Correlation (FRC)
            # frc_values = fourier_ring_correlation(original_fft[:len(wavelengths)],
            #                                       filtered_fft[:len(wavelengths)])
            frc_values.append(fourier_ring_correlation(original_fft[:len(wavelengths)],
                                                       filtered_fft[:len(wavelengths)]))
            # # Plot FRC
            # axs[2].plot(frc_values, color='red')

    # Store FRC values for multiple original spectra
    unfiltered_frc_values = []

    for spectra_index in spectra_indices[:-1]:
        spectrum_1_fft = np.fft.fft(x[spectra_index, :])  # Compute FFT of each spectrum
        spectrum_2_fft = np.fft.fft(x[spectra_index+1, :])  # Compute FFT of each spectrum

        unfiltered_frc = fourier_ring_correlation(spectrum_1_fft[:len(wavelengths)],
                                                  spectrum_2_fft[:len(wavelengths)])
        unfiltered_frc_values.append(unfiltered_frc)

    # Convert to NumPy array and compute the mean
    unfiltered_frc_values = np.array(unfiltered_frc_values)
    mean_unfiltered_frc = np.mean(unfiltered_frc_values, axis=0)

    # Plot both unfiltered and filtered FRC
    axs[2].plot(mean_unfiltered_frc, color='black', linestyle='dashed', label="Unfiltered FRC")

    print(len(frc_values))
    # Convert list to NumPy array for easier averaging
    frc_values = np.array(frc_values)  # Shape: (num_spectra, num_frequencies)

    # Compute mean FRC across all spectra
    mean_frc = np.mean(frc_values, axis=0)

    axs[2].plot(mean_frc, color=FRUIT_COLOR)
    axs[2].set_title("Fourier Ring Correlation (FRC) Between\nOriginal and Filtered Spectrum")
    axs[2].set_xlabel("Fourier Coordinates")
    axs[2].set_ylabel("FRC (Correlation strength)")

    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Counts")

    axs[1].set_xlabel("Fourier coordinate")
    axs[1].set_ylabel("Spectral Intensity (Fourier Domain)")
    plt.tight_layout()
    plt.savefig("FRC_reference.jpeg")
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
                label=f"λ = {wavelength} nm for background calculation")

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
    plt.savefig(f"tomato_reference_window.jpeg")
    plt.show()


if __name__ == '__main__':
    sg_params = {"window_length": 7, "polyorder": 2}
    # vis_data_and_fourier_transform("reference", range(60), sg_filter_params=sg_params)

    # plot_spectrum_histogram("reference", threshold=425+5*8.5)
    plot_spectrum_with_threshold("reference", 360)
