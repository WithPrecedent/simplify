
from datetime import timedelta
from dataclasses import dataclass

from more_itertools import unique_everseen
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


@dataclass
class Monograph(object):
    """Contains a single pandas dataframe or series, dictionary of column names
    and datatypes, and lists of columns by datatype.

    Attributes:
        df: pandas dataframe or series.
        types: a dictionary of column names and datatypes. In addition to
            official types (bool, float, int, object, CategoricalDtype, list,
            np.datetime64, timedelta), string labels for alternative
            types are included ('encoder', 'interactor', 'scaler').
    """

    df : object = None
    types : object = None

    def __post_init__(self):
        """Sets up default values and initial lists and dictionaries of column
        names and datatypes.
        """
        # Dictionary containing lists of columns with that datatype (or if
        # column was dropped, it is stored in 'dropped').
        self.type_lists = {bool : [],
                           float : [],
                           int : [],
                           object : [],
                           CategoricalDtype : [],
                           list : [],
                           np.datetime64 : [],
                           timedelta : [],
                           'interactor' : [],
                           'scaler' : [],
                           'encoder' : [],
                           'dropped' : []}
        # Creates default values for columns when data is missing.
        self.default_values = {bool : False,
                               float : 0.0,
                               int : 0,
                               object : '',
                               CategoricalDtype : '',
                               list : [],
                               np.datetime64 : 1/1/1900,
                               timedelta : 0,
                               'interactor' : '',
                               'scaler' : 0,
                               'encoder' : ''}
        # Initializes column lists and dict based upon arguments passed when
        # class is instanced.
        self.initialize()
        return self

    def __getitem__(self):
        """Returns df when __getitem__ accessed."""
        return self.df

    def __len__(self):
        """Returns length of df when __len__ accessed."""
        return len(self.df)

    def _add_cols(self, datatype, column_list):
        """Adds new columns to column lists and dict."""
        self.type_lists[datatype].extend(self._listify(column_list))
        self.types.update(dict.fromkeys(self._listify(column_list), datatype))
        return self

    def _apply_datatypes(self):
        for column_name, datatype in self.types.items():
            self.type_lists[datatype].append(column_name)
        return self

    def _deduplicate(self, type_list):
        """Removes duplicates from a list"""
        return list(unique_everseen(type_list))

    def _infer_datatypes(self):
        """Infers column datatypes and adds those datatypes to types."""
        self.types = {}
        for datatype, column_list in self.type_lists.items():
            if not datatype in ['interactor', 'scaler', 'encoder', 'dropped']:
                columns = self.df.select_dtypes(
                            include = [datatype]).columns.to_list()
                if columns:
                        self.type_lists[datatype].extend(columns)
                self.types.update(dict.fromkeys(columns, datatype))
        return self

    def _listify(self, column):
        """Checks to see if the columns are stored in a list. If not, the
        columns are converted to a list or a list of 'none' is created.
        """
        if not column:
            return ['none']
        elif isinstance(column, list):
            return column
        else:
            return [column]

    def _remove_from_list(self, column_list, new_columns):
        return [col for col in column_list if col not in new_columns]

    def _set_columns(self, df = None):
        """Creates attribute for columns specific to dataframe or series
        attribute. Format for new attribute is [df_name]_columns.
        """
        var_name = self._get_variable_name(df)
        if self.column_dict:
            setattr(self, var_name + '_columns',
                    Monograph(types = self.column_dict))
        else:
            setattr(self, var_name + '_columns', Monograph(df = df))

        return self

    @property
    def booleans(self):
        """Returns boolean columns."""
        return self._deduplicate(self.type_lists[bool])

    @property
    def categoricals(self):
        """Returns caterogical columns."""
        return self._deduplicate(self.type_lists[CategoricalDtype])

    @property
    def datetimes(self):
        """Returns datetime columns."""
        return self._deduplicate(self.type_lists[np.datetime64])

    @property
    def dropped(self):
        """Returns list of dropped columns."""
        return self._deduplicate(self.type_lists['dropped'])

    @property
    def encoders(self):
        """Returns columns with 'encoder' datatype."""
        return self._deduplicate(self.type_lists['encoder'])

    @property
    def floats(self):
        """Returns float columns."""
        return self._deduplicate(self.type_lists[float])

    @property
    def integers(self):
        """Returns int columns."""
        return self._deduplicate(self.type_lists[int])

    @property
    def interactors(self):
        """Returns columns with 'interactor' datatype."""
        return self._deduplicate(self.type_lists['interactor'])

    @property
    def lists(self):
        """Returns list columns."""
        return self._deduplicate(self.type_lists[list])

    @property
    def names(self):
        """Returns column names."""
        return self._deduplicate(list(self.types.keys()))

    @property
    def numerics(self):
        """Returns float and int columns."""
        return self._deduplicate(self.floats + self.integers)

    @property
    def scalers(self):
        """Returns columns with 'scaler' datatype."""
        return self._deduplicate(self.type_lists['scaler'])

    @property
    def strings(self):
        """Returns str (object type) columns."""
        return self._deduplicate(self.type_lists[object])

    @property
    def timedeltas(self):
        """Returns timedelata columns."""
        return self._deduplicate(self.type_lists[timedelta])

    def add_datatype(self, datatype, default_value = None):
        self.type_lists.update({datatype : []})
        self.default_values.update({datatype : default_value})
        return self

    def change_default(self, datatype, default_value):
        self.default_values[datatype] = default_value
        return self

    def crosscheck(self, df, column_names = None):
        for column_name in self._listify(column_names):
            if column_name in self.types:
                self.types.pop(column_name)
        for datatype, column_list in self.type_lists.items():
            if not datatype in ['dropped']:
                for column in column_list:
                    if not column in df.columns:
                        self.types[datatype].remove(column)
                if column_names:
                    self.types[datatype] = (
                            self._remove_from_list(
                                    column_list, self._listify(column_names)))
        return self

    def drop(self, df, column_names):
        if isinstance(column_names, str):
            self.types['dropped'].append(column_names)
        elif isinstance(column_names, list):
            self.types['dropped'].extend(column_names)
        self.crosscheck(df, column_names)
        return self

    def initialize(self):
        if self.types:
            self._apply_datatypes()
        elif (isinstance(self.df, pd.DataFrame)
                or isinstance(self.df, pd.Series)):
            self._infer_datatypes()
        else:
            error = 'Columns requires either types dict or df pandas object'
            raise AttributeError(error)
        return self