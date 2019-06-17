
from dataclasses import dataclass
import numpy as np
import os
import re

from .implement import Implement
from .matcher import ReMatch


@dataclass
class Combiner(Implement):
    """Class for combining data into new variables."""

    settings : object
    dicts_path: str
    section : str
    data_type : str
    munge_file : str

    def __post_init__(self):
        return self


    def _combine_list_all(self, df, in_columns, out_column):
        df[out_column] = np.where(np.all(df[self._listify(in_columns)]),
                                         True, False)
        return self

    def _combine_list_any(self, df, in_columns, out_column):
        df[out_column] = np.where(np.any(df[self._listify(in_columns)]),
                                         True, False)
        return self

    def _combine_list_dict(self, df, in_columns, out_column, combiner):
        df[out_column] = np.where(np.any(
                                df[self._listify(in_columns)]),
                                True, False)
        return self

    def combine(self, df, in_columns = None, out_column = None,
                combiner = None):
        combine_techniques = {'all' : self._combine_list_all,
                              'any' : self._combine_list_any}
        if isinstance(combiner, dict):
            self._combine_list_dict(df = df,
                                    in_columns = in_columns,
                                    out_column = out_column,
                                    combiner = combiner)
        else:
            combine_techniques[combiner](df = df,
                                        in_columns = in_columns,
                                        out_column = out_column)
        return self