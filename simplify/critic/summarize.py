"""
.. module:: summarize
:synopsis: summarizes data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.plan import SimplePlan
from simplify.core.step import SimpleStep


@dataclass
class Summarize(SimplePlan):
    """Summarizes data.

    Args:
        steps(dict(str: SimpleStep)): names and related SimpleStep classes for
            explaining data analysis models.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_read (bool): whether to call the 'read' method when the class
            is instanced.
    """
    steps: object = None
    name: str = 'summarizer'
    auto_publish: bool = True
    auto_read: bool = False
    lazy_import: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _read_export_parameters(self, file_name, file_format, transpose):
        self.export_parameters = {'folder': 'experiment',
                                  'file_name': file_name,
                                  'file_format': file_format}
        if not transpose:
            self.report = self.report.transpose()
            self.export_parameters.update({'header': False, 'index': True})
        else:
            self.export_parameters.update({'header': True, 'index': False})
        return self

    def _read_report(self, df):
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

    def _set_columns(self):
        self.columns = ['variable'] + (list(self.options.keys()))
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """Sets options for Summarize class."""
        super().draft()
        self.options = {
                'datatype': ['dtype'],
                'count': 'count',
                'min':'min',
                'q1': ['quantile', 0.25],
                'median': 'median',
                'q3': ['quantile', 0.75],
                'max': 'max',
                'mad': 'mad',
                'mean': 'mean',
                'stan_dev': 'std',
                'mode': ['mode', [0]],
                'sum': 'sum',
                'kurtosis': 'kurtosis',
                'skew': 'skew',
                'variance': 'var',
                'stan_error': 'sem',
                'unique': 'nunique'}
        return self

    def publish(self):
        self._set_columns()
        self.report = pd.DataFrame(columns = self.columns)
        return self

    def read(self, df = None, transpose = True, file_name = 'data_summary',
                file_format = 'csv'):
        """Creates a DataFrame of common summary data.

        Args:
            df(DataFrame): data to create summary report for.
            transpose(bool): whether the 'df' columns should be listed
                horizontally (True) or vertically (False) in 'report'.
            file_name(str): name of file to be exported (without extension).
            file_format(str): exported file format.
        """
        self._read_report(df = df)
        self._read_export_parameters(file_name = file_name,
                                        file_format = file_format,
                                        transpose = transpose)
        return self.report

