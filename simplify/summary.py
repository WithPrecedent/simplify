
from dataclasses import dataclass

import pandas as pd


@dataclass
class Summary(object):
    """Stores and exports a DataFrame of summary data for pandas DataFrame.

    Summary is more inclusive than pandas.describe() and includes
    boolean and numerical columns by default. It is also extensible: more
    metrics can easily be added to the report DataFrame.

    Parameters:
        auto_prepare: a boolean value indicating whether the prepare
            method should be automatically called.
    """
    auto_prepare : bool = True

    def __post_init__(self):
        self._set_defaults()
        if self.auto_prepare:
            self.prepare()
        return self

    def _set_defaults(self):
        """Sets options for Summary."""
        self.options = {'datatype' : ['dtype'],
                        'count' : 'count',
                        'min' :'min',
                        'q1' : ['quantile', 0.25],
                        'median' : 'median',
                        'q3' : ['quantile', 0.75],
                        'max' : 'max',
                        'mad' : 'mad',
                        'mean' : 'mean',
                        'stan_dev' : 'std',
                        'mode' : ['mode', [0]],
                        'sum' : 'sum',
                        'kurtosis' : 'kurtosis',
                        'skew' : 'skew',
                        'variance' : 'var',
                        'stan_error' : 'sem',
                        'unique' : 'nunique'}
        return self

    def prepare(self):
        """Prepares columns list for Summary report and initializes report."""
        self.columns = ['variable']
        self.columns.extend(list(self.options.keys()))
        self.report = pd.DataFrame(columns = self.columns)
        return self

    def start(self, df = None, transpose = True):
        """Completes report with data from df.

        Parameters:
            df: pandas DataFrame.
            transpose: boolean value indicating whether the df columns should be
                listed horizontally (True) or vertically (False) in report.
        """
        for i, column in enumerate(df.columns):
            row = pd.Series(index = self.columns)
            row['variable'] = column
            if df[column].dtype == bool:
                df[column] = df[column].astype(int)
            if df[column].dtype.kind in 'bifcu':
                for key, value in self.options.items():
                    if isinstance(value, str):
                        row[key] = getattr(df[column], value)()
                    elif isinstance(value, list):
                        if len(value) < 2:
                            row[key] = getattr(df[column], value[0])
                        elif isinstance(value[1], list):
                            row[key] = getattr(df[column],
                               value[0])()[value[1]]
                        else:
                            row[key] = getattr(df[column],
                               value[0])(value[1])
            self.report.loc[len(self.report)] = row
        if not transpose:
            self.report = self.report.transpose()
            self.df_header = False
            self.df_index = True
        else:
            self.df_header = True
            self.df_index = False
        return self