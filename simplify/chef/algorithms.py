"""
.. module:: chef algorithms
:synopsis: siMpLify machine learning algorithms
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.book import Book
from simplify.core.book import PageOutline


def auto_categorize(
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Union[List[str], str]] = None,
        threshold: int = 10) -> None:
    """Automatically assesses each column to determine if it has less than
    threshold unique values and is not boolean. If so, that column is
    converted to category type.

    Args:
        df (DataFrame): pandas object for columns to be evaluated for
            'categorical' type.
        columns (list or str): column names to be checked.
        threshold (int): number of unique values necessary to form a
            category. If there are less unique values than the threshold,
            the column is converted to a category type. Otherwise, it will
            remain its current datatype.

    Raises:
        KeyError: if a column in 'columns' is not in 'df'.

    """
    for column in self._check_columns(columns):
        try:
            if not column in self.booleans:
                if df[column].nunique() < threshold:
                    df[column] = df[column].astype('category')
                    self.columns[column] = 'categorical'
        except KeyError:
            error = ' '.join([column, 'is not in df'])
            raise KeyError(error)
    return self

# @make_columns_parameter
@backup_df
def convert_rare(self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0) -> None:
    """Converts categories rarely appearing within categorical columns
    to empty string if they appear below the passed threshold.

    The threshold is defined as the percentage of total rows.

    Args:
        df (DataFrame): pandas object with 'categorical' columns.
        columns (list): column names for datatypes to be checked. If it is
            not passed, all 'categorical' columns will be checked.
        threshold (float): indicates the percentage of values in rows
            below which a default value is substituted.

    Raises:
        KeyError: if column in 'columns' is not in 'df'.

    """
    if not columns:
        columns = self.categoricals
    for column in columns:
        try:
            df['value_freq'] = df[column].value_counts() / len(df[column])
            df[column] = np.where(
                df['value_freq'] <= threshold,
                self.default_values['categorical'],
                df[column])
        except KeyError:
            error = column + ' is not in df'
            raise KeyError(error)
    if 'value_freq' in df.columns:
        df.drop('value_freq', axis = 'columns', inplace = True)
    return self

# @make_columns_parameter
@backup_df
def decorrelate(self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0.95) -> None:
    """Drops all but one column from highly correlated groups of columns.

    The threshold is based upon the .corr() method in pandas. 'columns' can
    include any datatype accepted by .corr(). If 'columns' is None, all
    columns in the DataFrame are tested.

    Args:
        df (DataFrame): pandas object to be have highly correlated features
            removed.
        threshold (float): the level of correlation using pandas corr method
            above which a column is dropped. The default threshold is 0.95,
            consistent with a common p-value threshold used in social
            science research.

    """
    try:
        corr_matrix = df[columns].corr().abs()
    except TypeError:
        corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    corrs = [col for col in upper.corrs if any(upper[col] > threshold)]
    self.drop_columns(columns = corrs)
    return self

# @make_columns_parameter
@backup_df
def drop_infrequent(self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0) -> None:
    """Drops boolean columns that rarely are True.

    This differs from the sklearn VarianceThreshold class because it is only
    concerned with rare instances of True and not False. This enables
    users to set a different variance threshold for rarely appearing
    information. 'threshold' is defined as the percentage of total rows (and
    not the typical variance formulas used in sklearn).

    Args:
        df (DataFrame): pandas object for columns to checked for infrequent
            boolean True values.
        columns (list or str): columns to check.
        threshold (float): the percentage of True values in a boolean column
            that must exist for the column to be kept.
    """
    if columns is None:
        columns = self.booleans
    infrequents = []
    for column in listify(columns):
        try:
            if df[column].mean() < threshold:
                infrequents.append(column)
        except KeyError:
            error = ' '.join([column, 'is not in df'])
            raise KeyError(error)
    self.drop_columns(columns = infrequents)
    return self


# @make_columns_parameter
@backup_df
def smart_fill(self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[Union[List[str], str]] = None) -> None:
    """Fills na values in a DataFrame with defaults based upon the datatype
    listed in 'all_datatypes'.

    Args:
        df (DataFrame): pandas object for values to be filled
        columns (list): list of columns to fill missing values in. If no
            columns are passed, all columns are filled.

    Raises:
        KeyError: if column in 'columns' is not in 'df'.

    """
    for column in self._check_columns(columns):
        try:
            default_value = self.all_datatypes.default_values[
                    self.columns[column]]
            df[column].fillna(default_value, inplace = True)
        except KeyError:
            error = column + ' is not in DataFrame'
            raise KeyError(error)
    return self


def split_xy(self, label: Optional[str] = 'label') -> None:
    """Splits df into 'x' and 'y' based upon the label ('y' column) passed.

    Args:
        df (DataFrame): initial pandas object to be split.
        label (str or list): name of column(s) to be stored in 'y'.'

    """
    self.x = df[list(df.columns.values).remove(label)]
    self.y = df[label],
    self.label_datatype = self.columns[label]
    self._crosscheck_columns()
    self.state.change('train_test')
    return self
