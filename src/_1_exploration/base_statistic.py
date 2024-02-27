"""
Current module is used for building basics and statistics graphics
"""
import numpy
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as ppt


def generate_picture(column, counts_df):
    """
    Func for generating and saving plot picture as png
    :param column: str -> name of column im padas Series counts_df
    :param counts_df: pandas Series -> contain values for draw a plot
    """
    save_path = "graphics\\pre-proc\\"
    fig = ppt.Figure()
    ax = fig.add_subplot(111)
    ax.bar(dict(counts_df).keys(), dict(counts_df).values())
    ax.set_title(column)
    canvas = FigureCanvas(fig)
    canvas.print_figure(save_path + column + '.png')
    print(f"New image was created by path: {save_path + column + '.png'}")


def data_analysis(user_data, config):
    """
    Func for calc basic stat params (mean, meadian, STD, variance)
    :param user_data: pandas DataFrame with diff str variation
    :param config: config dict
    :return: base_statistics_df: pandas DtatFrame that contain basic stat params
    """
    # create new pandas Dataframe for collecting basic statistic parameters
    base_statistics_df = pd.DataFrame(columns=["Parameter", "Mean", "Median", "STD", "Variance"])

    # iterate by columns, calc a count of each option in each column
    # calc base statistics parameters
    # and packing values in new pandas DataFrame
    columns = user_data.columns
    for count, column in enumerate(columns):
        # calc a count of each option
        counts_df = user_data[column].value_counts(ascending=True)
        # creating and saving plot as png (draw if config)
        if config["is_draw_basic_stats"]:
            generate_picture(column, counts_df)

        # calc stat params for current column (mean, meadian, STD, variance)
        stat_params_per_column = pd.DataFrame([column, counts_df.mean(), counts_df.median(), counts_df.std(),
                                               counts_df.var(ddof=0)], ).transpose()

        # add calced params in main pandas df
        base_statistics_df = pd.DataFrame(numpy.concatenate(
            [base_statistics_df.values, stat_params_per_column.values],
            axis=0), columns=base_statistics_df.columns)

    return base_statistics_df
