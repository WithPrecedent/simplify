"""
explorer.base:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify


@dataclasses.dataclass
class Ledger(Book):
    """

    Args:


    """
    name: Optional[str] = 'explorer'
    steps: Optional[Dict[str, 'SimpleDirector']] = None

    def __post_init__(self) -> None:
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
        self._implement_report(df = recipe.dataset.df)
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
                for key, value in self.workers.items():
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
        self.columns = ['variable'] + (list(self.workers.keys()))
        self.report = pd.DataFrame(columns = self.columns)
        return self

    """ Core siMpLify methods """

    def draft(self) -> None:
        """Sets default options for the Explorer's analysis."""
        self._options = SimpleRepository(contents = {
            'summary': ('simplify.explorer.steps.summarize', 'Summarize'),
            'test': ('simplify.explorer.steps.test', 'Test')}
        # Sets plan container
        self.chapter_type = Chapter
        return self

    def publish(self, dataset: ['Dataset']) -> None:
        """
        """
        super().publish(data = dataset)
        return self


@dataclasses.dataclass
class Summary(Chapter):

    def __post_init__(self) -> None:
        super().__post_init__()
        return self