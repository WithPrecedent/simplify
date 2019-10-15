"""
.. module:: summarize
:synopsis: summarizes data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.critic.review import CriticTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'datatype': self._get_datatype,
    'count': ['numpy.ndarray', 'size'],
    'min':['numpy', 'nanmin'],
    'q1': ['numpy', 'nanquantile'],
    'median': ['numpy', 'nanmedian'],
    'q3': ['numpy', 'nanquantile'],
    'max': ['numpy', 'nanmax'],
    'mad': ['scipy.stats', 'median_absolute_deviation'],
    'mean': ['numpy', 'nanmean'],
    'std': ['numpy', 'nanstd'],
    'standard_error': ['scipy.stats', 'sem'],
    'geometric_mean': ['scipy.stats', 'gmean'],
    'geometric_std': ['scipy.stats', 'gstd'],
    'harmonic_mean': ['scipy.stats', 'hmean'],
    'mode': ['scipy.stats', 'mode'],
    'sum': ['numpy', 'nansum'],
    'kurtosis': ['scipy.stats', 'kurtosis'],
    'skew': ['scipy.stats', 'skew'],
    'variance': ['numpy', 'nanvar'],
    'variation': ['scipy.stats', 'variation'],
    'unique': ['numpy', 'nunique']}


@dataclass
class Summarize(CriticTechnique):
    """Summarizes data.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """
    technique: object = None
    parameters: object = None
    name: str = 'summary'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _implement(self, recipe = None, transpose = True,
                  file_name = 'data_report', file_format = 'csv'):
        """Creates a DataFrame of common report data.

        Args:
            df(DataFrame): data to create report report for.
            transpose(bool): whether the 'df' columns should be listed
                horizontally (True) or vertically (False) in 'report'.
            file_name(str): name of file to be exported (without extension).
            file_format(str): exported file format.
        """
        self._implement_report(df = recipe.ingredients.df)
        self._implement_export_parameters(file_name = file_name,
                                          file_format = file_format,
                                          transpose = transpose)
        return self

    def _implement_export_parameters(self, file_name, file_format, transpose):
        self.export_parameters = {
            'folder': 'experiment',
            'file_name': file_name,
            'file_format': file_format}
        if not transpose:
            self.report = self.report.transpose()
            self.export_parameters.update({'header': False, 'index': True})
        else:
            self.export_parameters.update({'header': True, 'index': False})
        return self

    def _implement_report(self, df):
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

    def _start_report(self):
        self.columns = ['variable'] + (list(self.options.keys()))
        self.report = pd.DataFrame(columns = self.columns)
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """Sets options for Summarize class."""
        super().draft()
        self.extra_parameters = {
            'kurtosis': {'nan_policy': 'omit'},
            'mad': {'nan_policy': 'omit'},
            'mode': {'nan_policy': 'omit'},
            'q1': {'q': 0.25},
            'q3': {'q': 0.75},
            'sem': {'nan_policy': 'omit'},
            'skew': {'nan_policy': 'omit'},
            'variation': {'nan_policy': 'omit'}}
        self.simplify_options = ['datatype']
        return self

    def publish(self):
        self._start_report()
        super().publish()
        return self

