from itertools import combinations
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regressors import stats
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

import webscrape

__all__ = ['RidgeRegression', 'LassoRegression']


# directory of data
_DATA_DIR = os.path.join(os.path.dirname(webscrape.__file__), "data")


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
    filepath = os.path.join(_DATA_DIR, filepath)

    # import data for stat and year from csv file into dataframe
    if not os.path.exists(filepath):
        webscrape.get_stat_data(stat, year=year)

    return pd.read_csv(filepath_or_buffer=filepath)


def create_df(main_stat, other_stats, year=None):
    """For a given dependent stat and a list of independent stats, returns a
    pd.DataFrame instance where the dependent stat is in the first column and
    predictors are in subsequent columns

    Parameters
    ----------
    main_stat : str
        Dependent stat.
    other_stats : str or list
        Independent stat(s). Can pass a single stat as a str or a list of str
        for multiple stats.
    year : int, optional
        Year of specified statistic. If None, then it defaults to the most
        recent (including current) season.

    Returns
    -------
    df_combined : pd.DataFrame
        DataFrame of rankings for each stat. The dependent stat is in the first
        columns and predictors are in subsequent columns
    """

    # if only one independent stat, turn it into list
    if isinstance(other_stats, str):
        other_stats = [other_stats]

    # check that the main stat isn't in the other stats
    if main_stat in other_stats:
        raise ValueError("The '{}' main stat cannot be in other stats".format(main_stat))

    # get data from csv files
    df_main = get_from_csv(main_stat, year=year)
    df_others = []
    for stat in other_stats:
        df_combined = get_from_csv(stat, year=year)
        df_combined = df_combined.set_index("PLAYER NAME")
        df_combined = df_combined.reindex(index=df_main["PLAYER NAME"])
        df_combined = df_combined.reset_index()
        df_others.append(df_combined)

    # create y and x arrays
    y = np.array(df_main["RANK THIS WEEK"])
    x = []
    for df_combined in df_others:
        x.append(np.array(df_combined["RANK THIS WEEK"]))
    x = np.column_stack(tuple(x))

    # create filter that removes rows with NaN in x and corresponding rows in y
    if len(x.shape) == 1:
        nan_filter = ~np.isnan(x)
    else:
        nan_filter = np.all(~np.isnan(x), axis=1)
    x = x[nan_filter]
    y = y[nan_filter]

    # combine x and y in a pd.DataFrame instance
    df_combined = pd.DataFrame(np.column_stack((y, x)), columns=[main_stat] + other_stats)

    return df_combined


class AnalysisBase(object):

    def __init__(self, main_stat, other_stats, year=None):
        self.main_stat = main_stat
        self.other_stats = other_stats
        self.year = year
        self.df = create_df(main_stat, other_stats, year=year)
        self.x_mean = np.zeros(self.df.shape[1] - 1)
        self.x_std = np.ones(self.df.shape[1] - 1)
        self.y_mean = 0

    def get_X(self, as_numpy=True):
        X = self.df.iloc[:, 1:]
        if as_numpy:
            return np.array(X)
        else:
            return X

    def get_y(self):
        y = self.df.iloc[:, 0]
        return y

    def _resave_df(self, X, y):
        self.df = pd.DataFrame(np.column_stack((y, X)), columns=[self.main_stat] + list(self.other_stats))

    def normalize_X(self):
        X = self.get_X()
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        X = (X - self.x_mean) / self.x_std
        self._resave_df(X, self.get_y())

    def center_y(self):
        y = self.get_y()
        self.y_mean = np.mean(y)
        y = y - self.y_mean
        self._resave_df(self.get_X(), y)

    def fit(self, **kwargs):
        pass

    def fit_multi(self, *args, **kwargs):
        pass

    def variance_inflation_factor(self):
        return [variance_inflation_factor(self.get_X(), i) for i in range(self.df.shape[1] - 1)]


class LassoRegression(AnalysisBase):

    REGRESSION_TYPE = Lasso
    REGRESSION_TYPE_CV = LassoCV

    def __init__(self, main_stat, other_stats, year=None):
        super(LassoRegression, self).__init__(main_stat, other_stats, year=year)
        self.alpha = np.array([])
        self.coef = None
        self.mse = np.array([])
        self.mse_std = np.array([])
        self.best_coef = None
        self.best_mse = None
        self.CV = RepeatedKFold(n_splits=self.get_X().shape[0], n_repeats=1, random_state=2652124)

    def fit(self, alpha):
        model = self.REGRESSION_TYPE(alpha=alpha)
        model.fit(self.get_X(), self.get_y())
        coef = model.coef_.reshape(len(model.coef_), 1)
        self.alpha = np.append(self.alpha, alpha)
        if self.coef is None:
            self.coef = coef
        else:
            self.coef = np.column_stack((self.coef, coef))
        self.compute_mse(alpha)

    def compute_mse(self, alpha):
        model = self.REGRESSION_TYPE(alpha=alpha)
        X = self.get_X()
        y = self.get_y()
        scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=self.CV, n_jobs=-1)
        scores = np.abs(scores)
        self.mse = np.append(self.mse, np.mean(scores))
        self.mse_std = np.append(self.mse, np.std(scores))

    def fit_multi(self, alphas):
        self.alpha = np.array([])
        self.coef = None
        for alpha in alphas:
            self.fit(alpha)

    def plot_coef(self):
        plt.figure()
        plt.plot(self.alpha, self.coef.T)
        plt.xscale('log')
        plt.xlabel(r"$\lambda$")
        plt.ylabel("Standardized coefficients")
        plt.legend(labels=self.other_stats)
        plt.tight_layout()

    def plot_mse(self):
        plt.figure()
        plt.plot(self.alpha, self.mse)
        plt.xscale('log')
        plt.xlabel(r"$\lambda$")
        plt.ylabel("MSE")
        plt.tight_layout()

    def fit_CV(self, alphas):
        search = self.REGRESSION_TYPE_CV(alphas=alphas, scoring='neg_mean_squared_error', cv=self.CV)
        results = search.fit(self.get_X(), self.get_y())
        self.best_mse = np.abs(results.best_score_)
        self.best_coef = results.coef_
        self.best_model = search

    def summary(self):
        stats.summary(self.best_model, self.get_X(), self.get_y())


class RidgeRegression(LassoRegression):

    REGRESSION_TYPE = Ridge
    REGRESSION_TYPE_CV = RidgeCV


def best_subset(estimator, X, y, max_size=8, cv=5):
    """Calculates the best model of up to max_size features of X.
       estimator must have a fit and score functions.
       X must be a DataFrame."""

    n_features = X.shape[1]
    subsets = (combinations(range(n_features), k + 1)
               for k in range(min(n_features, max_size)))

    best_size_subset = []
    for subsets_k in subsets:  # for each list of subsets of the same size
        best_score = -np.inf
        best_subset = None
        for subset in subsets_k: # for each subset
            estimator.fit(X.iloc[:, list(subset)], y)
            # get the subset with the best score among subsets of the same size
            score = estimator.score(X.iloc[:, list(subset)], y)
            if score > best_score:
                best_score, best_subset = score, subset
        # to compare subsets of different sizes we must use CV
        # first store the best subset of each size
        best_size_subset.append(best_subset)

    # compare best subsets of each size
    best_score = -np.inf
    best_subset = None
    list_scores = []
    for subset in best_size_subset:
        score = cross_val_score(estimator, X.iloc[:, list(subset)], y, cv=cv, scoring='neg_mean_squared_error').mean()
        list_scores.append(score)
        if score > best_score:
            best_score, best_subset = score, subset

    return best_subset, best_score, best_size_subset, list_scores


if __name__ == '__main__':
    pass

