
from dataclasses import dataclass

import pandas as pd

from simplify.core.step import Step


@dataclass
class Summarize(Step):
    """Stores and exports a DataFrame of summary data for pandas DataFrame.

    Summary is more inclusive than pandas.describe() and includes
    boolean and numerical columns by default. It is also extensible: more
    metrics can easily be added to the report DataFrame.

    Args:
        name: a string designating the name of the class which should be
            identical to the section of the idea configuration with relevant
            settings.
        auto_finalize: a boolean value that sets whether the finalize method is
            automatically called when the class is instanced.
        auto_produce: sets whether to automatically call the 'produce' method
            when the class is instanced.
    """
    name : str = 'summarizer'
    auto_finalize : bool = True
    auto_produce : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        """Sets options for Summarize class."""
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

    def finalize(self):
        """Prepares columns list for Summary report and initializes report."""
        self.columns = ['variable']
        self.columns.extend(list(self.options.keys()))
        self.report = pd.DataFrame(columns = self.columns)
        return self

    def _produce_report(self, df = None, transpose = True):
        """Completes report with data from df.

        Args:
            df: pandas DataFrame.
            transpose: boolean value indicating whether the df columns should be
                listed horizontally (True) or vertically (False) in report.
        """
        for column in df.columns:
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

    def produce(self, df = None, transpose = True, file_name = 'data_summary',
              file_format = 'csv'):
        """Creates and exports a DataFrame of common summary data using the
        Summary class.

        Args:
            df: a pandas DataFrame.
            transpose: boolean value indicating whether the df columns should
                be listed horizontally (True) or vertically (False) in report.
            file_name: string containing name of file to be exported.
            file_format: string of file extension from Depot.extensions.
        """
        self._produce_report(df = df, transpose = transpose)
        if self.verbose:
            print('Saving summary data')
        self.depot.save(variable = self._produce_report.report,
                            folder = self.depot.experiment,
                            file_name = file_name,
                            file_format = file_format,
                            header = self._produce_report.df_header,
                            index = self._produce_report.df_index)
        return self