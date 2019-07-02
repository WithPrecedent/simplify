
from dataclasses import dataclass
import numpy as np

from .stage import Stage, Technique


@dataclass
class Bundle(Stage):
    """Class for combining different datasets."""

    technique : str = ''
    parameters : object = None
    name : str = 'clean'
    auto_prepare : bool = True

    def __post_init__(self):
        self.techniques = {'mappers' : Mapper,
                           'mergers' : Merger}
        super().__post_init__()
        return self

    def start(self, df, sources):
        for technique_name in self.technique_names:
            for section, technique in getattr(self, technique_name).items():
                df, sources[section] = technique(df, sources[section])
        return df, sources

@dataclass
class Mapper(Technique):

    source_column : str = ''
    out_column : str = ''
    mapper : object = None

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

    def start(self, df):
        return df

@dataclass
class Merger(Technique):

    index_columns : object = None
    merge_type : str = ''

    def __post_init__(self):
        return self

    def start(self, dfs):
        merged_df = None
        return merged_df