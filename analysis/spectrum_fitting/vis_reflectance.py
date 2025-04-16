# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from functools import lru_cache

# installed libraries
from matplotlib.colors import to_rgb, to_hex
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# plt.style.use('seaborn-v0_8-dark')
# local files
import get_data


def darken_color(hex_color, amount=30):
    rgb = [int(c * 255) for c in to_rgb(hex_color)]
    darker_rgb = [max(c - amount, 0) for c in rgb]
    return to_hex([c / 255 for c in darker_rgb])


COLOR_MAP_LYCOPENE = ["palegreen", "red"]
# COLOR_MAP_BETA = ["#fffcc9", "#ffe066", "#ff9900", "#cc3300"]
COLOR_MAP_BETA_START = ["#f0f0c0", "#ffe066", "#ff9900", "#cc3300"]
COLOR_MAP_BETA = [darken_color(c, amount=15) for c in COLOR_MAP_BETA_START]

ALPHA = 0.5

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


@lru_cache()  # this gets called 5-7 times so just memorize it instead of making a class
def make_color_map(color_min: float, color_max: float, target: str = "lycopene"
                   ) -> tuple[mpl.colors.LinearSegmentedColormap,
                              mpl.colors.Normalize]:
    """ Make a linear segment color map from a list

    Args:
        color_min (float): Minimum value for colormap normalization.
        color_max (float): Maximum value for colormap normalization.
        target (str): Name of the component to visualize, lycopene or carotene,
        The color will be selected by the target

    Returns:
        tuple: A tuple containing:
            - mpl.colors.LinearSegmentedColormap: Linear segmented colormap object.
            - mpl.colors.Normalize: Normalization object for the colormap.

        Examples:
            x, y = get_data.get_x_y(sensor="as7262", leaf="mango", measurement_type="reflectance")
            lines = plt.plot(x, y)
            color_map, map_norm = make_color_map(y["Avg Total Chlorophyll (µg/cm2)"].min(),
                                                 y["Avg Total Chlorophyll (µg/cm2)"].max())
            for i, line in enumerate(lines):  # type: int, mpl.lines.Line2D
                line.set_color(color_map(map_norm(y.iloc[i])))

    """

    if "lycopene" in target:
        color_map = COLOR_MAP_LYCOPENE
    elif "carotene" in target:
        color_map = COLOR_MAP_BETA
    else:
        raise ValueError(f"Color map needs 'lycopene' or 'carotene', '{target}' is not valid")

    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "", color_map)
    map_norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
    return color_map, map_norm


def vis_single_sensor(ax: plt.Axes = None,
                      sensor: str = "as7262",
                      fruit: str = "tomato",
                      measurement_type: str = "reflectance",
                      target="lycopene (DW)",
                      led: str = "b'White",
                      **kwargs):

    if ax is None:  # for testing
        _, ax = plt.subplots(1, 1)
    if sensor == "as7265x":  # set the LED
        kwargs["led"] = led
    x, y, groups = get_data.get_data(sensor=sensor,
                                     fruit=fruit,
                                     measurement_mode=measurement_type,
                                     target_column=target,
                                     mean_spot=True,
                                     **kwargs)

    color_map, map_norm = make_color_map(y.min(), y.max(), target=target)
    print(x.columns)
    if x.columns.dtype != "float64":
        x_values = [float(col.split(' ')[0]) for col in x.columns]
    else:
        x_values = x.columns
    print(x_values)
    lines = ax.plot(x_values, x.T, alpha=ALPHA, lw=1.5)
    for i, line in enumerate(lines):  # type: int, mpl.lines.Line2D
        line.set_color(color_map(map_norm(y.iloc[i])))
    return mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map)


def vis_4_sensors(fruit: str, target: str, sensor_settings: dict):
    figure, axes = plt.subplots(4, 1, figsize=(5, 7), sharex=True)
    for i, sensor in enumerate(["as7262", "as7263", "as7265x", "c12880"]):
        print(i, sensor)
        color_map = vis_single_sensor(ax=axes[i], sensor=sensor,
                                      target=target, fruit=fruit,
                                      led="b'White'",
                                      **sensor_settings)
    plt.subplots_adjust(right=0.83)
    COLOR_BAR_AXIS = [.89, .13, 0.02, 0.8]
    color_bar_axis = figure.add_axes(COLOR_BAR_AXIS)
    if fruit == "tomato":
        color_bar_label = "Total Lycopene (mg/100 g)"
    elif fruit == "mango":
        color_bar_label = "Beta Carotene (mg/100 g)"
    else:
        raise ValueError(f"Choose a fruit mango or tomato, {fruit} is not valid")

    figure.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                    label=color_bar_label, fraction=0.08)
    # make x axis label at bottom of graph
    axes[3].set_xlabel("Wavelength (nm)")
    # make y-axis label in the middle to cover all graphs
    figure.text(0.03, 0.5, 'Normalized Reflectance',
                ha='center', va='center', rotation='vertical', fontsize=12)
    figure.suptitle(f"{fruit.capitalize()} Reflectance")
    # figure.tight_layout()
    figure.subplots_adjust(left=0.11, right=0.87, wspace=0.19, top=0.95)
    plt.show()


if __name__ == '__main__':
    sensor_settings = {"led current": "12.5 mA",
                       "integration time": 50}
    # vis_single_sensor(sensor="as7262", target="beta-carotene (DW)",
    #                   **sensor_settings)
    # vis_single_sensor(sensor="c12880", target="beta-carotene (DW)",
    #                   **sensor_settings)
    # vis_single_sensor(
    #     sensor="as7265x", target="lycopene (FW)",
    #     fruit='tomato', led="b'White'",
    #     **sensor_settings)
    # plt.show()
    vis_4_sensors("mango", "carotene (FW)",
                  sensor_settings)
    # vis_4_sensors("tomato", "lycopene (FW)",
    #               sensor_settings)
