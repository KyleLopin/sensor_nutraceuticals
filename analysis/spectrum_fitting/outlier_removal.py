# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler

# local files
import get_data
from vis_reflectance import make_color_map

SENSORS = ["as7262", "as7263", "as7265x"]
SENSORS = ["c12880"]
MANGO_TARGETS = ('phenols (FW)', 'phenols (DW)', 'carotene (FW)', 'carotene (DW)', '%DM')
TOMATO_TARGETS = ('%DM', 'lycopene (DW)', 'lycopene (FW)',
                  'beta-carotene (DW)', 'beta-carotene (FW)')
GROUPS_COLUMNS = ["Fruit", "spot", "Read number"]
logs = []


def calculate_residues(x: pd.DataFrame,
                       groups: pd.Series) -> pd.DataFrame:
    """
    Calculate residuals for each row in a DataFrame based on the difference of the individual
    value(s) and the average of all the other rows in that group.

    Args:
        x (pandas.DataFrame): Input DataFrame containing the data.
        groups (pandas.Series): Series defining the groups for each row in the DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing residuals for each row,
        calculated based on group-wise means.
        The index and columns of the returned DataFrame match those of the input DataFrame 'x'.
    """
    if isinstance(x, pd.DataFrame):
        new_df = pd.DataFrame(columns=x.columns)
    elif isinstance(x, pd.Series):
        new_df = pd.Series()
    else:
        raise ValueError(f"x has to be a pd.Series or pd.DataFrame, not a {type(x)}")
    for index in x.index:
        # get indexes with the same leaf number minus the current index
        # print(index)
        # print(groups)
        # print(groups.loc[index])
        fruit, spot, read_number = groups.loc[index]
        # other_leaf_indexes = groups.index[groups == groups[index]].drop(index)
        # Find all other rows with same fruit and spot (but different read number)
        mask = ((groups["Fruit"] == fruit) &
                (groups["spot"] == spot))
        # calculate the difference between the current index and the mean of the
        # spectrum for the other reads on same fruit / spot
        other_spot_indexes = groups.index[mask].drop(index)
        # print("other indexes: ", other_spot_indexes)
        residue = x.loc[index] - x.loc[other_spot_indexes].mean()
        # make new DataFrame with the residues
        new_df.loc[index] = residue
    return new_df


def mahalanobis_outlier_removal(x: pd.DataFrame,
                                groups: pd.DataFrame,
                                use_robust: bool = True,
                                display_hist: bool = False,
                                cutoff_limit: float = 3.0,
                                ) -> np.ndarray[bool]:
    """
    Remove outliers from a DataFrame based on Mahalanobis distance.

    Parameters:
        x (pd.DataFrame): Input DataFrame containing numerical data.
        use_robust (bool): Whether to use a robust estimator (default True).
        display_hist (bool): Whether to display histograms (default False).
        If True the function will also show the data in a blocking way.
        cutoff_limit (float): Number of standard deviations to consider outliers (default 3.0).

    Returns:
        pd.DataFrame: DataFrame mask
    """
    # Standardize data before Mahalanobis distance
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if use_robust:  # fit a MCD robust estimator to data
        covariance = MinCovDet()
    else:  # fit a MLE estimator to data
        covariance = EmpiricalCovariance().fit(x)
    # print(x)
    # print(x[x.isna().any(axis=1)])
    covariance.fit(x)
    # calculate the mahalanobis distances
    # and calculate the cubic root to simulate a normal distribution
    mahal_distances = covariance.mahalanobis(x - covariance.location_)**0.33
    # shift the distribution to calculate the distance from the standard deviation easier
    shifted_mahal = mahal_distances - mahal_distances.mean()
    cutoff = cutoff_limit*shifted_mahal.std()
    # Data-level mask (e.g., based on Mahalanobis cutoff)
    data_mask = np.where((-cutoff < shifted_mahal) & (shifted_mahal < cutoff), True, False)
    # Group-level mask (e.g., remove groups with >= 2 outliers)
    outlier_mask = pd.Series(~data_mask, index = groups.index)
    # now remove spots that have 2 or more outliers
    group_outlier_counts = outlier_mask.groupby([groups["Fruit"], groups["spot"]]).sum()
    groups_to_remove = group_outlier_counts[group_outlier_counts > 2].index
    # Map group mask to row level
    remove_mask = groups.set_index(["Fruit", "spot"]).index.isin(groups_to_remove)
    # We want to keep rows that are in `data_mask == True` AND not in `remove_mask`
    data_mask = data_mask & (~remove_mask)
    print(f"group outlier counts: {group_outlier_counts.shape}")
    print(f"groups to remove: {groups_to_remove}")
    # print(f"index to remove: {np.where(remove_mask)[0]}")
    # print(f"index to remove2: {np.where(~data_mask)[0]}")

    if display_hist:
        plt.hist(shifted_mahal, bins=100)
        plt.axvline(cutoff, ls='--')
        print(f"outliers removed = {x.shape[0]-x[data_mask].shape[0]}")
        plt.show()
    print(f"outliers removed = {x.shape[0] - x[data_mask].shape[0]}")
    logs.append(f"outliers removed = {x.shape[0] - x[data_mask].shape[0]}")
    # print(f"data mask: {data_mask}")
    if False: #TODO change to get indices flag
        outlier_indexes = np.where(~data_mask)[0]
    return data_mask


def save_mean_data(sensor: str,
                   fruit: str,
                   measurement_type: str = "reflectance",
                   residue_threshold: float = 3.0):
    """
    Load spectral data, remove outliers using Mahalanobis distance on residuals,
    average by ["Fruit", "spot"], and save combined result to a CSV file.

    Parameters
    ----------
    sensor : str
        Sensor name (e.g., 'as7262', 'as7263', 'as7265x', 'c12880').

    fruit : str
        Fruit name (e.g., 'tomato', 'mango').

    measurement_type : str, default="reflectance"
        Type of measurement: "raw", "reflectance", or "absorbance".

    residue_threshold : float, default=3.0
        Cutoff used in Mahalanobis outlier removal.
    """
    # Step 1: Build sensor settings
    # sensor_settings = {
    #     "led current": "12.5 mA",
    #     "integration time": 50
    # }
    sensor_settings = {}
    if sensor == "as7265x":
        sensor_settings["led"] = "b'White IR'"

    targets = TOMATO_TARGETS
    if fruit == "mango":
        targets = MANGO_TARGETS

    # Step 2: Load the data
    # x, y, groups = get_data.get_data(sensor=sensor,
    #                                  fruit=fruit,
    #                                  measurement_mode=measurement_type,
    #                                  target_column=targets,
    #                                  **sensor_settings)

    data = get_data.get_data(sensor=sensor,
                             fruit=fruit,
                             measurement_mode=measurement_type,
                             target_column=targets,
                             split_x_y_groups=False)
    print("data: ",  data)
    # 2. Group by integration time and led current
    setting_groups = data.groupby(["integration time", "led current"])
    print(setting_groups.groups.keys())
    x_columns = [col for col in data.columns if " nm" in col]
    print(x_columns)
    final_df = pd.DataFrame()

    for (int_time, led_current), group in setting_groups:
        # Select subset manually using boolean masking
        sub_data = data[
            (data["integration time"] == int_time) &
            (data["led current"] == led_current)
            ]
        group_index = sub_data.index
        # print("settings: ", int_time, led_current)
        logs.append(f"settings: {int_time}, {led_current}")
        # print(sub_data.groupby(["integration time", "led current"]).nunique())
        counts = (
            sub_data.groupby(["integration time", "led current", "Fruit number", "spot"])
            .size()
            .reset_index(name="count")
        )
        filtered = counts[counts["count"] <= 2]
        # remove items with less than 2 reads, could be due to too few unsaturated reads
        small_combos = filtered.set_index(
            ["integration time", "led current", "Fruit number", "spot"]).index
        # print("filtered: ", filtered)
        data_index = sub_data.set_index(
            ["integration time", "led current", "Fruit number", "spot"]).index
        keep_mask = ~data_index.isin(small_combos)
        filtered_data = sub_data[keep_mask].reset_index(drop=True)
        residues = calculate_residues(filtered_data[x_columns],
                                      filtered_data[GROUPS_COLUMNS])
        mask = mahalanobis_outlier_removal(residues,
                                           filtered_data[GROUPS_COLUMNS],
                                           cutoff_limit=residue_threshold)
        masked_data = (filtered_data[mask].reset_index(drop=True)
                       .drop(columns=["Read number", "Unnamed: 0", "Fruit number"]))
        # Count remaining unique (Fruit number, spot) combos
        remaining_unique_combos = (
            masked_data[["Fruit", "spot"]]
            .drop_duplicates()
            .shape[0]
        )
        remaining_fruits = (
            masked_data[["Fruit"]]
            .drop_duplicates()
            .shape[0]
        )
        # print(masked_data)
        # print(masked_data.columns)
        print(f"Remaining unique (Fruit number, spot) combos: {remaining_unique_combos} "
              f"and fruit: {remaining_fruits}")
        logs.append(f"Remaining unique (Fruit number, spot) combos: {remaining_unique_combos} "
                    f"and fruit: {remaining_fruits}")
        agg = {}
        for column in masked_data:
            agg[column] = "mean"
        for column in ["integration time", "led current", "led", "saturation check", "time"]:
            agg[column] = "first"
        agg.pop("spot")
        agg.pop("Fruit")
        # print(masked_data.dtypes)
        # print(masked_data["spot"].unique())
        meaned = (masked_data.groupby(["Fruit", "spot"]).agg(agg)
                  .reset_index())
        for col in ["integration time", "gain"]:
            meaned[col] = meaned[col].astype(int)
        # print(meaned)
        # Step 5: Combine and save
        final_df = pd.concat([final_df, meaned], axis=0)
        print(final_df)
    print(final_df)
    for log in logs:
        print(log)

    output_file = f"{fruit}_{sensor}_mean_data.csv"
    meaned.to_csv(output_file)
    print(f"Saved cleaned mean data to: {output_file}")


def make_manuscript_figure(fruit: str = "mango"):
    """
    Create a figure comparing outliers detected by Mahalanobis distance and
    Mahalanobis distance on the residues across multiple sensors.

    Args:
        fruit (str): name of the fruit to get the data of

    """

    color_map, map_norm = make_color_map(0, 100)

    # =========== INNER FUNCTION
    def make_outlier_plot(x: pd.DataFrame, y: pd.Series,
                          use_residue: bool, ax: plt.Axes,
                          groups: pd.Series = None):
        """
        Generate a plot to visualize data and highlight outliers.

        Parameters:
        ----------
        x : pd.DataFrame
            Feature data where each row corresponds to a sample and each column
            represents a wavelength or feature.
        y : pd.Series
            Target values (e.g., labels or intensities) corresponding to each sample in `x`.
        use_residue : bool
            If True, calculate outliers based on Mahalanobis distance of residuals.
            If False, calculate outliers directly from `x`.
        ax : plt.Axes
            Matplotlib Axes object to plot the data.
        groups : pd.Series, optional
            Grouping information for samples, used when calculating residuals.
            Defaults to None.

        Notes:
        -----
        - Outliers are highlighted in red, while other data points are colored
          based on their target values using a colormap.
        - Residuals are calculated if `use_residue` is True, and groups are provided.

        Returns:
        -------
        None
            The plot is drawn on the provided `ax` object.
        """
        print("using residue: ", use_residue, )
        if use_residue:
            # print(x.loc[632:692])
            # print(groups.loc[632:692])
            # print("+++++", fruit, sensor)
            residues = calculate_residues(x, groups)
            # print(residues.loc[632:692])
            mask = mahalanobis_outlier_removal(residues, groups)
        else:
            mask = mahalanobis_outlier_removal(x, groups)
        if sensor == "c12880":
            x_wavelengths = x.columns
        else:
            wavelengths = x.columns
            x_wavelengths = [int(wavelength.split()[0]) for wavelength in wavelengths]
        print('leaves left =', groups[mask].nunique())

        for idx in range(x.shape[0]):
            if mask[idx]:
                color, alpha, z = color_map(map_norm(y.iloc[idx])), 0.3, 1
            else:
                # Determine the final color using the colormap based on y
                color, alpha, z = 'red', 0.8, 2

            ax.plot(x_wavelengths, x.iloc[idx, :],
                    color=color, alpha=alpha, zorder=z, lw=1)
    # = ========== END INNER FUNCTION

    # Create a figure
    fig = plt.figure(figsize=(6, 8))

    # Create GridSpec with 4 rows and 2 columns
    gs = GridSpec(5, 2,
                  figure=fig, height_ratios=[1, 1, 0.2, 1, 1])

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[3, :])  # Merge row 3, columns 0 and 1
    ax6 = fig.add_subplot(gs[4, :])  # Merge row 3, columns 0 and 1

    for sensor in SENSORS:
        sensor_settings = {"led current": "12.5 mA",
                           "integration time": 50,
                           "led": "White LED"}
        # led = "White LED"
        # int_time = 50
        target = "lycopene (FW)"
        if fruit == "mango":
            target = "carotene (FW)"
        if sensor == "as7265x":
            sensor_settings["led"] = "b'White IR'"
        elif sensor == "as7263":
            sensor_settings["integration time"] = 150
        # get the data
        x, y, groups = get_data.get_data(fruit=fruit, sensor=sensor,
                                         target_column=target,
                                         measurement_mode="reflectance",
                                         split_x_y_groups=True,
                                         **sensor_settings)
        # y = y["Avg Total Chlorophyll (µg/cm2)"]
        print(x.columns)
        if sensor == "c12880":
            x_wavelengths = x.columns
        else:
            wavelengths = [w.split()[0] for w in x.columns]
            x_wavelengths = [int(wavelength.split()[0]) for wavelength in wavelengths]

        if sensor == "as7262":  # DRY yourself, it's late
            # graph outliers based on Mahalanobis distances only
            make_outlier_plot(x, y, use_residue=False, ax=ax1, groups=groups)
            ax1.set_xticks(ticks=x_wavelengths, labels=[])
            make_outlier_plot(x, y, use_residue=True, ax=ax2, groups=groups)
            ax2.set_xticks(ticks=x_wavelengths, labels=wavelengths, rotation=35)
            ax2.set_xlabel("Wavelengths (nm)", fontsize=12)

        elif sensor == 'as7263':
            make_outlier_plot(x, y, use_residue=False, ax=ax3, groups=groups)
            ax3.set_xticks(ticks=x_wavelengths, labels=[])
            make_outlier_plot(x, y, use_residue=True, ax=ax4, groups=groups)
            ax4.set_xticks(ticks=x_wavelengths, labels=wavelengths, rotation=35)
            ax4.set_xlabel("Wavelengths (nm)", fontsize=12)

        elif sensor == 'c12880':
            make_outlier_plot(x, y, use_residue=False, ax=ax3, groups=groups)
            make_outlier_plot(x, y, use_residue=True, ax=ax4, groups=groups)

        elif sensor == "as7265x":
            cutoff = 1
            # print("applying cutoff")
            # print(x[cutoff:])
            # print(type(x))
            make_outlier_plot(x.iloc[:, cutoff:], y, use_residue=False,
                              ax=ax5, groups=groups)
            ax5.set_xticks(ticks=x_wavelengths[cutoff:], labels=[])
            make_outlier_plot(x.iloc[:, cutoff:], y, use_residue=True,
                              ax=ax6, groups=groups)
            ax6.set_xticks(ticks=x_wavelengths[cutoff:],
                           labels=wavelengths[cutoff:], rotation=60)
            ax6.set_xlabel("Wavelengths (nm)", fontsize=12)
        else:
            raise ValueError(f"Incorrect sensor passed")

    # set titles
    ax1.set_title("AS7262")
    ax3.set_title("AS7263")
    ax5.set_title("AS7265x", pad=-10)
    # set y limits
    ax1.set_ylim([0.05, 0.55])
    ax2.set_ylim([0.05, 0.55])
    ax3.set_ylim([0.05, 1.1])
    ax4.set_ylim([0.05, 1.1])

    # label to each axis with a-f
    for i, (ax, letter) in enumerate(zip([ax1, ax2, ax3, ax4, ax5, ax6],
                                         ['a', 'b', 'c', 'd', 'e', 'f'])):
        # weird \n are to align all text on top
        condition = "Outliers\n(Spectrum)"
        coords = [(0.01, .70), (0.12, .75)]
        if i in [4, 5]:
            coords = [(0.03, .70), (0.08, .75)]
        if i % 2 == 1:  # odd numbers are calculated by residues
            condition = "Outliers\n(Residues)"
        ax.annotate(f"({letter})\n", coords[0], xycoords='axes fraction',
                    fontsize=11, fontweight='bold')
        ax.annotate(condition, coords[1], xycoords='axes fraction',
                    fontsize=10)

    ax5.plot([], [], color='red', label="Outlier")
    ax5.legend(loc="lower right")
    # add color bar at end
    color_map = mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map)
    color_bar_axis = fig.add_axes([.90, .1, 0.02, 0.8])
    color_bar = fig.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                             fraction=0.08)
    # add y-axis labels
    fig.text(0.46, 0.72, 'Normalized Reflectance',
             ha='center', va='center', rotation='vertical', fontsize=12)
    fig.text(0.03, 0.71, 'Normalized Reflectance',
             ha='center', va='center', rotation='vertical', fontsize=12)
    fig.text(0.03, 0.28, 'Normalized Reflectance',
             ha='center', va='center', rotation='vertical', fontsize=12)

    # Adjust the label padding (distance from the color bar)
    color_bar.set_label(r'Total Chlorophyll ($\mu$g/cm$^2$)',
                        labelpad=-5, fontsize=12)
    fig.subplots_adjust(left=0.1, wspace=0.26, right=0.86, hspace=0.24)

    # plt.tight_layout()
    fig.savefig("outlier_detection.jpeg", dpi=600)
    plt.show()


if __name__ == '__main__':
    save_mean_data(sensor="c12880", fruit="tomato")

    # make_manuscript_figure("tomato")
    #
    # sensor = "c12880"
    # fruit = "tomato"
    # measurement_type = "reflectance"
    # # led = "White LED"
    # sensor_settings = {"led current": "12.5 mA",
    #                    "integration time": 50}
    # if sensor == "as7265x":
    #     sensor_settings["led"] = "b'White IR'"
    # x, y, groups = get_data.get_data(sensor=sensor,
    #                                  fruit=fruit,
    #                                  measurement_mode=measurement_type,
    #                                  mean_spot=False,
    #                                  target_column='lycopene (FW)',
    #                                  **sensor_settings)
    # residues = calculate_residues(x, groups)
    # mahalanobis_outlier_removal(residues, display_hist=True)
