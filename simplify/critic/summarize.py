
from dataclasses import dataclass

import pandas as pd

from simplify.core.base import SimplePlan, SimpleStep


@dataclass
class Summarize(SimplePlan):
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
    """

    steps : object = None
    name : str = 'summarizer'
    auto_finalize : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _get_columns(self):
        """Returns columns list for 'report' DataFrame."""
        return ['variable'] + (list(self.options.keys()))

    def _produce_export_parameters(self, file_name, file_format, transpose):
        self.export_parameters = {'folder' : 'experiment',
                                  'file_name' : file_name,
                                  'file_format' : file_format}
        if not transpose:
            self.report = self.report.transpose()
            self.export_parameters.update({'header' : False, 'index' : True})
        else:
            self.export_parameters.update({'header' : True, 'index' : False})
        return self

    def _produce_report(self, df):
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
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """Sets options for Summarize class."""
        super().draft()
        self.options = {
                'full': FullSummary,
                'describe': Describe}
        return self

    def finalize(self):
        self.report = pd.DataFrame(columns = self._get_columns())
        return self

    def produce(self, df = None, transpose = True, file_name = 'data_summary',
                file_format = 'csv'):
        """Creates a DataFrame of common summary data.

        Args:
            df (DataFrame): data to create summary report for.
            transpose (bool): whether the 'df' columns should be listed
                horizontally (True) or vertically (False) in 'report'.
            file_name (str): name of file to be exported (without extension).
            file_format (str): exported file format.
        """
        self._produce_report(df = df)
        self._produce_export_parameters(file_name = file_name,
                                        file_format = file_format,
                                        transpose = transpose)
        return self


@dataclass
class Describe(SimpleStep):

    def draft(self):
        self.options = {}
        return self


@dataclass
class FullSummary(SimpleStep):

    def draft(self):
        self.options = {
                'datatype' : ['dtype'],
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
