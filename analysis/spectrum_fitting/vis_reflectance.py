# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from functools import lru_cache

# installed libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# local files
import get_data

COLOR_MAP_LYCOPENE = ["palegreen", "red"]
COLOR_MAP_BETA = ["palegreen", "goldenrod"]
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


@lru_cache()  # this gets called 5-7 times so just memorize it instead of making a class
def make_color_map(color_min: float, color_max: float, target: str="lycopene"
                   ) -> tuple[mpl.colors.LinearSegmentedColormap,
                              mpl.colors.Normalize]:
    """ Make a linear segment color map from a list

    Args:
        color_min (float): Minimum value for colormap normalization.
        color_max (float): Maximum value for colormap normalization.

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
                line.set_color(tuple(color_map(map_norm(y["Avg Total Chlorophyll (µg/cm2)"]))[i]))

    """

    if "lycopene" in target:
        color_map = COLOR_MAP_LYCOPENE
    elif "carotene" in target:
        color_map = COLOR_MAP_BETA

    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "", color_map)
    map_norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
    return color_map, map_norm


def vis_single_sensor(ax: plt.Axes = None,
                      sensor: str = "as7262",
                      fruit: str = "tomato",
                      measurement_type: str = "reflectance",
                      target="lycopene (DW)",
                      led: str = "b'White IR'",
                      **kwargs):

    if ax is None:  # for testing
        _, ax = plt.subplots(1, 1)
    if sensor == "as7265x":  # set the led
        kwargs["led"] = led
    x, y, groups = get_data.get_data(sensor=sensor,
                                     fruit=fruit,
                                     measurement_mode=measurement_type,
                                     target_column=target,
                                     mean_spot=True,
                                     **kwargs)
    # data = get_data.get_data(sensor=sensor,
    #                          fruit=fruit,
    #                          measurement_mode=measurement_type,
    #                          target_column=target,
    #                          mean_spot=True,
    #                          split_x_y_groups=False,
    #                          **kwargs)

    color_map, map_norm = make_color_map(y.min(), y.max(), target=target)
    # print(data)
    print(x.shape)
    print(kwargs)
    lines = ax.plot(x.T, alpha=0.3)
    for i, line in enumerate(lines):  # type: int, Line2D
        line.set_color(color_map(map_norm(y.iloc[i])))
    plt.show()


if __name__ == '__main__':
    sensor_settings = {"led current": "12.5 mA",
                       "integration time": 50}
    # vis_single_sensor(sensor="as7262", target="beta-carotene (DW)",
    #                   **sensor_settings)
    vis_single_sensor(sensor="c12880", target="beta-carotene (DW)",
                      **sensor_settings)
