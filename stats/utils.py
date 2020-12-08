import os

import numpy as np
import pandas as pd

from data import webscrape


# directory of data
_PGA_DATA_DIR = os.path.join(os.path.dirname(webscrape.__file__), "pga_data")


#############
# UTILITIES #
#############


def get_from_csv(stat, year=None):
    """Imports data for specified stat and year into a pandas DataFrame. If csv
    file doesn't exist, then it will scrape the web for the data and save it to
    a csv file.

    Parameters
    ----------
    stat : str
        Name of the stat. Case doesn't matter.
    year : int, optional
        Year of specified statistic. If None, then it defaults to the most
        recent (including current) season.
    """

    # create filepath for stat and year
    filepath = "_".join(stat.split(" "))
    if year is not None:
        filepath += "_{}".format(year)
    filepath += ".csv"
    filepath = os.path.join(_PGA_DATA_DIR, filepath)

    # import data for stat and year from csv file into dataframe
    if not os.path.exists(filepath):
        webscrape.get_stat_data(stat, year=year)

    return pd.read_csv(filepath_or_buffer=filepath)


def create_yx(main_stat, other_stats, year=None):
    """For a given dependent stat and a list of independent stats, returns a
    vector for the dependent stat and an array for the independent stat.

    Parameters
    ----------
    main_stat : str
        Dependent stat.
    other_stats : str or list
        Independent stat(s). Can pass a single stat as a str or a list of str
        for multiple stats.
    """

    # check that the main stat isn't in the other stats
    if main_stat in other_stats:
        raise ValueError("The '{}' main stat cannot be in other stats".format(main_stat))

    # get data from csv files
    df_main = get_from_csv(main_stat, year=year)
    df_others = []
    for stat in other_stats:
        df = get_from_csv(stat, year=year)
        df = df.set_index("PLAYER NAME")
        df = df.reindex(index=df_main["PLAYER NAME"])
        df = df.reset_index()
        df_others.append(df)

    # create y and x arrays
    y = np.array(df_main["RANK THIS WEEK"])
    x = None
    for df in df_others:
        if x is None:
            x = np.array(df["RANK THIS WEEK"])
        else:
            x = np.vstack((x, np.array(df["RANK THIS WEEK"])))
    x = x.T

    # remove rows with NaN in x and corresponding rows in y
    if len(x.shape) == 1:
        nan_filter = ~np.isnan(x)
    else:
        nan_filter = np.all(~np.isnan(x), axis=1)
    x = x[nan_filter]
    y = y[nan_filter]

    return y, x


if __name__ == '__main__':
    pass
