
from dataclasses import dataclass
import numpy as np
import pandas as pd
import re

from more_itertools import unique_everseen

@dataclass
class ReTool(object):
    """Contains shared methods for regex tools in the ReTool package.

    ReTool aims to simplify and speed up creating expressions tables (pseudo-
    dictionaries) for use with pandas dataframes and series. This can
    particularly help data scientists munging text data with keywords instead
    of natural language processing.

    Because the methods iterrate through the expressions dictionaries, any
    efficiency gains of vectorization in the target dataframes or series are
    lost with very long lookup tables. Further, because regular expressions
    cannot be hashed like ordinary dictionary keys, some of the speed
    advantages of dictionaries cannot be replicated. The tipping point for
    expressions dataframe length versus using .apply or other non-vectorized
    options for matching varies and needs to be tested based upon the
    particular use case. The normal use case where ReMatch has efficiency gains
    is with a very large dataframe and a relatively small (< 500 rows)
    expressions dataframe.
    """
    def __post_init__(self):
        self._initialize()
        return self

    @property
    def default_value(self):
        """Returns default value for appropriate datatype."""
        return self.default_values(self.out_type)

    @property
    def matcher(self):
        """Returns matcher method for appropriate datatype."""
        return getattr(self, '_' + self.out_type)

    def _aggregate_flags(self, row):
        """Creates regex flag sequence for compiling regexes."""
        self.flags = None
        for name, flag in self.all_flags.items():
            if name in row.index and row[name]:
                if not self.flags:
                    self.flags = flag
                else:
                    self.flags |= flag
        return self

    def _build_expressions(self):
        """Builds regular expressions table."""
        self.expressions = pd.DataFrame(list(zip(self.keys, self.values)),
                                        columns = ['keys', 'values'])
        # Implements common column naming scheme regardless of source of data
        # for the expressions table.
        self.keys = 'keys'
        self.values = 'values'
        # If user selects to compile the regular expressions, this section
        # includes columns in the raw table for flags selected.
        if self.compile_keys:
            for flag in self.all_flags.keys():
                if self.flags and flag in self.flags:
                    self.expressions[flag] = True
        return self

    def _check_out_column(self, out_column):
        """Checks if out_column passed. If not, default is used."""
        if out_column:
            out_column = self.out_column
        return out_column

    def _compile_expressions(self, key_value):
        """Compiles regular expressions (whether built or loaded)."""
        for i, row in self.expressions.iterrows():
            self._aggregate_flags(row)
            if self.flags:
                self.expressions.loc[i, key_value] = (
                    re.compile(self.expressions.loc[i, key_value],
                               flags = self.flags))
            else:
                self.expressions.loc[i, self.keys] = (
                    re.compile(self.expressions.loc[i, key_value]))
        return self

    def _convert_to_dict(self):
        self.expressions = self.expressions[[self.keys, self.values]]
        self.expressions = (self.expressions.set_index(
                self.keys).to_dict()[self.values])
        return self

    def _initialize(self):
        self._set_defaults()
        if self.file_path:
            self._load_expressions()
        else:
            if isinstance(self.keys, list) or isinstance(self.keys, pd.Series):
                if (isinstance(self.values, list)
                        or isinstance(self.values, pd.Series)):
                    self._build_expressions()
                else:
                    error = 'values must be list or series if no file_path'
                    raise TypeError(error)
            else:
                error = 'keys must be list or series if no file_path'
                raise TypeError(error)
        # Calls method to convert loaded and passed data to expressions table.
        self._make_expressions()
        return self

    def _load_expressions(self):
        """Loads data for expressions table from .csv file, converts keys to
        strings, and removes a common encording error character.
        """
        self.expressions = (pd.read_csv(self.file_path,
                                        index_col = False,
                                        encoding = self.encoding,
                                        true_values = ['y', 'Y', '1'],
                                        false_values = ['n', 'N', '0'])
                              .astype(str)
                              .replace('Â', ''))
        return self

    def _make_expressions(self):
        if hasattr(self, 'complie_keys') and self.compile_keys:
            self._compile_expressions(self.keys)
        if hasattr(self, 'complie_values') and self.compile_values:
            self._compile_expressions(self.values)
        return self

    def _set_defaults(self):
        # Sets default values for missing data based upon datatype of column.
#        if not self.default_values:
        self.default_values = {'bool' : False,
                               'float' : 0.0,
                               'int' : 0,
                               'list' : [],
                               'pattern' : '',
                               'patterns' : [],
                               'str' : ''}
        self.all_flags = {'ignorecase' : re.IGNORECASE,
                          'dotall' : re.DOTALL,
                          'locale' : re.LOCALE,
                          'multiline' : re.MULTILINE,
                          'verbose' : re.VERBOSE,
                          'ascii' : re.ASCII}
        if not self.encoding:
            self.encoding = 'windows-1252'
        return self

    def update(self, key, value):
        self.expressions.update({key : value})
        return self

@dataclass
class ReFrame(ReTool):
    """Stores and applies vectorized string and regular expression matching
    methods (when possible) to pandas dataframes.

    ReFrame allows keys and values to be formed from pandas series,
    python lists, or imported from a .csv file.

    Using the match method, the user can pass a dataframe containing source
    and output columns. Regular expressions are used as keys and may be
    compliled with or without any flag selected.

    If out_type is bool, new dataframe columns are created with headers derived
    from the values in the dictionary. A boolean value is returned.

    if out_type is 'pattern', a single column is used or created with the name
    passed in 'out_column.' The return is the matched values from the
    regular expression expressions table.

    If out_type is str, int or float, a single column is used or created
    with the header name passed in 'out_column.' The return is the matched
    value of the key in the expressions table.

    If out_type is list, a single column is used or created with the header
    name passed in 'out_column.' The return is all matched patterns based upon
    a regular expression stored in a python list within each dataframe column.
    """

    keys : str = 'keys'
    values : str = 'values'
    file_path : str = ''
    encoding : str = ''
    compile_keys : bool = True
    compile_values : bool = False
    source : str = ''
    source_suffix : str = ''
    out_column : str = ''
    out_suffix : str = ''
    flags : object = None
    out_type : str = 'bool'

    def __post_init__(self):
        super().__post_init__()
        self._convert_to_dict()
        return self

    def _bool(self, df):
        for key, value in self.expressions.items():
            df[value] = (np.where(df[self.source].str.contains(key, True,
                                  self.default_value)))
        return df

    def _float(self, df):
        df = self._str(df)
        df[self.out_column] = pd.to_numeric(df[self.out_column],
                                            errors = 'coerce',
                                            downcast = float)
        return df

    def _int(self, df):
        df = self._str(df)
        df[self.out_column] = pd.to_numeric(df[self.out_column],
                                            errors = 'coerce',
                                            downcast = 'integer')
        return df

    def _list(self, df):
        df[self.out_column] = np.empty((len(df), 0)).tolist()
        df = df.apply(self._list_row, axis = 'columns')
        return df

    def _list_row(self, df_row):
        temp_list = []
        for key, value in self.expressions.items():
            temp_list += re.findall(key, df_row[self.source])
        if temp_list:
            temp_list = list(unique_everseen(temp_list))
            df_row[self.out_column].extend(temp_list)
        return df_row

    def _pattern(self, df):
        for key, value in self.expressions.items():
            df[self.out_column] = df[self.source].str.find(key)
        return df

    def _patterns(self, df):
        for key, value in self.expressions.items():
            df[value] = df[self.source].str.findall(key)
        return df

    def _str(self, df):
        for i, row in self.expressions.iterrows():
            df[self.out_column] = (
                np.where(df[self.source].str.contains(row[self.keys]),
                         row[self.values], df[self.out_column]))
        return df

    def match(self, df, source = '', source_suffix = '', out_column = '',
              out_suffix = ''):
        if source:
            self.source = source
        if source_suffix:
            self.source_suffix = source_suffix
        if out_column:
            self.out_column = out_column
        if out_suffix:
            self.out_suffix = out_suffix
        df = self.matcher(df)
        return df

class ReSearch(ReTool):

    keys : str = 'keys'
    values : str = 'values'
    file_path : str = ''
    encoding : str = ''
    compile_keys : bool = True
    flags : object = None
    out_type : str = 'bool'
    out_prefix : str = ''

    def __post_init__(self):
        super().__post_init__()
        return self

    def _bool(self, df, regex):
        for key, value in self.expressions.items():
            out_column = self.out_prefix + value
            if re.search(regex, self.source):
                df[out_column] = True
            else:
                df[out_column] = False
        return df

    def _float(self, df, regex):
        for key, value in self.expressions.items():
            out_column = self.out_prefix + value
            if re.search(key, self.source):
                df[out_column] = value
                break
            else:
                df[out_column] = self.default_value
        df[out_column] = float(df[self.out_column])
        return df

    def _int(self, df, regex):
        for key, value in self.expressions.items():
            out_column = self.out_prefix + value
            if re.search(key, self.source):
                df[out_column] = value
                break
            else:
                df[out_column] = self.default_value
        df[out_column] = int(df[self.out_column])
        return df

    def _list(self, df, regex):
        temp_list = []
        for key, value in self.expressions.items():
            out_column = self.out_prefix + value
        temp_list += re.findall(regex, self.source)
        if temp_list:
            temp_list = list(unique_everseen(temp_list))
            df[out_column] = temp_list
        else:
            df[out_column] = self.default_value
        return df

    def _pattern(self, df, regex):
        for key, value in self.expressions.items():
            out_column = self.out_prefix + value
            if re.search(key, self.source):
                df[out_column] = value
                break
        return df

    def _patterns(self, df, regex):
        for key, value in self.expressions.items():
            out_column = self.out_prefix + value
            if re.search(key, self.source):
                df[out_column] = re.search(key, self.source).group(0).strip()
        return df

    def _str(self, df, regex):
        for key, value in self.expressions.items():
            out_column = self.out_prefix + value
            if re.search(key, self.source):
                df[out_column] = value
                break
            else:
                df[out_column] = self.default_value
        df[out_column] = str(df[self.out_column])
        return df

    def match(self, df, source, source_in_df = False):
        if source_in_df:
            self.source = df[source]
        else:
            self.source = source
        df = self.matcher(df = df)
        return df

class ReOrganize(ReTool):
    """Stores and applies string and regular expression matching methods to
    python strings for dividing strings into a pandas series.

    ReArrange allows keys and values to be formed from pandas series,
    python lists, or imported from a .csv file.

    Using the arrange method, the user should pass the source string and the
    pandas series (df, despite it not actually being a dataframe) and both will
    be returned.

    Regular expressions are stored as keys with the values representing the
    out_columns. Regular expressions may be compliled with or without any flag
    selected.

    """
    keys : str = 'keys'
    values : str = 'values'
    file_path : str = ''
    encoding : str = ''
    compile_keys : bool = True
    flags : object = None
    out_prefix : str = ''

    def __post_init__(self):
        super().__post_init__()
        self._convert_to_dict()
        return self

    def match(self, df, source, remove_from_source = True):
        for key, value in self.expressions.items():
            out_column = self.out_prefix + value
            if re.search(key, source):
                df[out_column] = re.search(key, source).group(0).strip()
                if remove_from_source:
                    source = re.sub(key, '', source)
            else:
                df[out_column] = self.default_values['str']
        return df, source