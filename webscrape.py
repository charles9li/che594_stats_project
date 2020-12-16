from collections import OrderedDict
import os
import unicodedata

from bs4 import BeautifulSoup
import requests

import pandas as pd


# directory of this module
_THIS_MODULE_DIR = os.path.dirname(__file__)


##################
# STAT PAGE URLs #
##################

# PGA Tour assigns each stat an index
_STAT_INDICES = {'driving distance':                101,
                 'driving accuracy percentage':     102,
                 'greens in regulation percentage': 103,
                 'putting average':                 104,
                 'official money':                  109,
                 'sand save percentage':            111,
                 'scrambling':                      130,
                 'proximity to hole':               331,
                 'sg: putting':                     '02564'}


def get_stat_url(stat, year=None):
    """Returns URL of web page with the specified stat and year.

    Parameters
    ----------
    stat : str
        Name of the stat. Case doesn't matter.
    year : int, optional
        Year of specified statistic. If None, then it defaults to the most
        recent (including current) season.

    Returns
    -------
    url : str
        URL of web page with the specified stat and year.
    """
    stat_index = _STAT_INDICES[stat.lower()]
    url = "https://www.pgatour.com/stats/stat.{}".format(stat_index)
    if year is not None:
        url += ".y{}".format(str(year))
    url += ".html"
    return url


##################
# HTML UTILITIES #
##################


def save_html(html, path):
    """Saves HTML to specified filepath.

    Parameters
    ----------
    html : bytes
        HTML tree to be saved.
    path : str
        Filepath to where HTML tree is saved.
    """
    with open(path, 'wb') as f:
        f.write(html)


def open_html(path):
    """Reads HTML tree from file and returns it.

    Parameters
    ----------
    path : str
        Filepath where HTML tree is saved.

    Returns
    -------
    html : bytes
        HTML tree.
    """
    with open(path, 'rb') as f:
        return f.read()


############
# SCRAPING #
############


def get_stat_data(stat, year=None):
    """Parses HTML tree for a given stat and year and saves to a csv file.

    Parameters
    ----------
    stat : str
        Name of the stat. Case doesn't matter.
    year : int, optional
        Year of specified statistic. If None, then it defaults to the most
        recent (including current) season.
    """

    # create html directory if it doesn't exist
    html_dir = os.path.join(_THIS_MODULE_DIR, "html")
    if not os.path.exists(html_dir):
        os.mkdir(html_dir)

    # path of HTML file for stat web page
    html_path = "html/{}".format("_".join(stat.lower().split(" ")))
    if year is not None:
        html_path += "_{}".format(str(year))
    html_path = os.path.join(_THIS_MODULE_DIR, html_path)

    # if HTML file doesn't exist, request web page and save HTML to file
    if not os.path.exists(html_path):
        url = get_stat_url(stat, year=year)
        r = requests.get(url)
        save_html(r.content, html_path)

    # retrieve HTML from file
    html = open_html(html_path)

    # create soup
    soup = BeautifulSoup(html, 'html.parser')

    # initialize data
    data = OrderedDict()

    # get table headers
    cols = soup.select('thead th')
    for col in cols:
        col_name = unicodedata.normalize('NFKD', col.text).replace('\n', '').strip()
        if col_name.startswith("AVG"):
            col_name = col_name.replace(".", "")
        data[col_name] = []

    # get data from each row
    rows = soup.select('tbody tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        cols = cleanup_row(cols, stat.lower(), list(data.keys()))
        for ele, col in zip(cols, data.keys()):
            data[col].append(ele)

    # create DataFrame
    df = pd.DataFrame(data=data)

    # save DataFrame to csv
    csv_path = "data/{}".format("_".join(stat.lower().split(" ")))
    if year is not None:
        csv_path += "_{}".format(str(year))
    csv_path += ".csv"
    csv_path = os.path.join(_THIS_MODULE_DIR, csv_path)
    df.to_csv(path_or_buf=csv_path, index=False)


_NEED_CONVERT = ['proximity to hole']


def cleanup_row(row, stat, headers):
    """Cleans up row by
        1. removing 'T' from ranks
        2. convert distance measurements from apostrophe-double prime notation
           to float

    Parameters
    ----------
    row : list
        A single row in the data table.
    stat : str
        Name of the stat. Case doesn't matter.
    headers : list
        List of column names.

    Returns
    -------
    row : list
        Fixed row.
    """

    # remove 'T' from ranks
    for col in ['RANK THIS WEEK', 'RANK LAST WEEK']:
        index = headers.index(col)
        row[index] = row[index].replace("T", "")

    # convert distance to float if necessary
    if stat.lower() in _NEED_CONVERT:
        index = headers.index('AVG')
        distance = row[index].replace('"', '').split("\'")
        distance = float(distance[0]) + float(distance[1]) / 12.
        row[index] = str(distance)

    return row


if __name__ == '__main__':
    pass
