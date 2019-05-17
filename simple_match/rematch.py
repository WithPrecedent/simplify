"""
Class for using vectorized string matching methods (when possible) across
pandas dataframes and series with regular expressions and ordinary strings as
keys.

ReMatch aims to simplify and speed up creating expressions tables (pseudo-
dictionaries) for use with pandas dataframes and series. This can particularly
help data scientists munging text data with keywords instead of natural
language processing.

Because the methods iterrate through the expressions dataframes, any efficiency
gains of vectorization in the target dataframes or series are lost with very
long lookup dataframes. Further, because regular expressions cannot be hashed
like ordinary dictionary keys, some of the speed advantages of dictionaries
cannot be replicated. The tipping point for expressions dataframe length versus
using .apply or other non-vectorized options for matching varies and needs to
be tested based upon the particular use case. The normal use case where ReMatch
has efficiency gains is with a very large dataframe and a relatively small
(< 500 rows) expressions dataframe.

ReMatch allows keys and values to be formed from pandas series, python lists,
or imported from a .csv file into a pandas dataframe.

Using the match method, the user can either:
    1) Pass a string (within a pandas series or freestanding) to find a match
    and store the result in a pandas series;
        or;
    2) Pass a series (dataframe column) of strings and store the result in
    one or more pandas series (dataframe column).

prefix and suffix parameters allow for iterables to be added to column or
index names in the dataframe.

Regular expressions are ordinarily used only as keys, but not values, unless
the reverse_dict option is set to True. Regular expressions may be compliled
with or without any flag selected.

If out_type is bool, new dataframe columns are created with headers derived
from the values in the dictionary. A boolean value is returned.

if out_type is 'pattern', a single column is used or created with the header
name passed in 'out_col.' The return is the matched values from the regular
expression expressions table.

If out_type is str, int or float, a single column is used or created
with the header name passed in 'out_col.' The return is the matched value of
the key in the expressions table.

If out_type is list, a single column is used or created with theheader name
passed in 'out_col.' The return is all matched patterns based upon a regular
expression stored in a python list within each dataframe or series cell.

If knots are provided, the passed dictionary should be a dataframe with
the key and knot columns being strings or regular expressions. The
returned value must be boolean and will be True if the key is matched but
the knot column is not matched.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import re

from more_itertools import unique_everseen


@dataclass
class ReMatch(object):

    keys : str = 'keys'
    values : str = 'values'
    reverse_dict : bool = False
    file_path : str = ''
    encoding : str = 'Windows-1252'
    compile_keys : bool = True
    flags : object = None
    out_type : str = bool
    in_col : str = ''
    in_prefix : str = ''
    in_suffix : str = ''
    out_col : str = ''
    out_prefix : str = ''
    out_suffix : str = ''
    convert_lists : bool = False
    default_value : str = ''
    expressions : object = None

    def __post_init__(self):
        self._store()
        """
        Creates expressions table from either .csv or passed arguments.
        """
        if self.file_path:
            self._load_expressions()
        else:
            if isinstance(self.keys, list) or isinstance(self.keys, pd.series):
                if (isinstance(self.values, list)
                        or isinstance(self.values, pd.series)):
                    self._build_expressions()
                else:
                    error = 'values must be list or series if no file_path'
                    raise TypeError(error)
            else:
                error = 'keys  must be list or series if no file_path'
                raise TypeError(error)
        """
        Calls method to convert loaded and passed data to expressions table.
        """
        self._make_expressions()
        return self

    def __get_item__(self, key):
        return self.expressions[key]

    def __set_item__(self, key, value):
        self.expressions.update({key, value})
        return self

    def __del_item__(self, key):
        self.expressions.pop(key)
        return self

    def _complete_col_name(self, in_col = '', out_col = ''):
        if in_col:
            return self.in_prefix + in_col + self.in_suffix
        if out_col:
            return self.out_prefix + out_col + self.out_suffix

    def _store(self):
        self.col_vars = ['in_col', 'in_prefix', 'in_suffix', 'out_col',
                         'out_prefix', 'out_suffix', 'default_value']
        for var in self.col_vars:
            setattr(self, 'stored_' + var, getattr(self, var))
        self.stored_in_col = self._complete_col_name(in_col = self.in_col)
        self.stored_out_col = self._complete_col_name(out_col = self.out_col)
        return self

    def _load_expressions(self):
        """
        Loads data for expressions table from .csv file, converts keys to
        strings, and removes a common encording error character.
        """
        self.expressions = (pd.read_csv(self.file_path,
                                   index_col = False,
                                   encoding = self.encoding,
                                   true_values = ['y', 'Y', '1'],
                                   false_values = ['n', 'N', '0'])
                         .astype(dtype = {self.keys : str})
                         .replace('Â', ''))
        return self

    def _build_expressions(self):
        """
        keys and values are zipped into pandas dataframe.
        """
        zip_data = list(zip(self.keys, self.values))
        self.expressions = pd.DataFrame(zip_data, columns = ['keys', 'values'])
        """
        Implements common column naming scheme regardless of source of data
        for the expressions table.
        """
        self.keys = 'keys'
        self.values = 'values'
        """
        If user selects to compile the regular expressions, this section
        includes columns in the raw table for flags selected.
        """
        self.all_flags = ['ignorecase', 'dotall', 'locale', 'multiline',
                          'verbose', 'ascii']
        if self.compile_keys:
            for flag in self.all_flags:
                if getattr(self, flag):
                    self.expressions[flag] = True
        return self

    def _compile_expressions(self):
        for i, row in self.expressions.iterrows():
            flags = self._aggregate_flags(row)
            if flags:
                self.expressions.loc[i, self.keys] = (
                    re.compile(self.expressions.loc[i, self.keys],
                               flags = flags))
            else:
                self.expressions.loc[i, self.keys] = (
                    re.compile(self.expressions.loc[i, self.keys]))
        return self

    def _aggregate_flags(self, row):
        flags = None
        first_flag = False
        if 'ignorecase' in row.index and row['ignorecase']:
            flags = re.IGNORECASE
            first_flag = True
        if 'dotall' in row.index and row['dotall']:
            if first_flag:
                flags |= re.DOTALL
            else:
                flags = re.DOTALL
                first_flag = True
        if 'locale' in row.index and row['locale']:
            if first_flag:
                flags |= re.LOCALE
            else:
                flags = re.LOCALE
                first_flag = True
        if 'multiline' in row.index and row['multiline']:
            if first_flag:
                flags |= re.MULTILINE
            else:
                flags = re.MULTILINE
                first_flag = True
        if 'verbose' in row.index and row['verbose']:
            if first_flag:
                flags |= re.VERBOSE
            else:
                flags = re.VERBOSE
                first_flag = True
        if 'ascii' in row.index and row['ascii']:
            if first_flag:
                flags |= re.ASCII
            else:
                flags = re.ASCII
        return flags

    def _make_expressions(self):
        if self.out_type in [bool, 'patterns']:
            self.expressions[self.values] = (self.out_prefix
                                             + self.expressions[self.values]
                                             + self.out_suffix)
        elif self.out_type in [list, str, int, float, 'pattern']:
            self.out_col = (self.out_prefix
                            + self.out_col
                            + self.out_suffix)
        if self.compile_keys:
            self._compile_expressions()
        self.expressions = self.expressions[[self.keys, self.values]]
        if self.reverse_dict:
            self.expressions = (self.expressions.set_index(
                    self.values).to_dict()[self.keys])
        else:
            self.expressions = (self.expressions.set_index(
                    self.keys).to_dict()[self.values])
        return self

    def _list_to_string(i_list):
        return ', '.join(i_list)

    def _bool_df_match(self, df):
        for key, value in self.expressions.items():
            df[value] = (np.where(df[self.in_col].str.contains(key, True,
                                  False)))
        return df

    def _bool_series_match(self, df, in_string):
        for key, value in self.expressions.items():
            if re.search(key, in_string):
                df[value] = True
            else:
                df[value] = False
        return df

    def _bool_match(self, df, in_string, is_df):
        if is_df:
            df = self._bool_df_match(df)
        else:
            df = self._bool_series_match(df, in_string)
        return df

    def _str_df_match(self, df):
        for i, row in self.expressions.iterrows():
            df[self.out_col] = (
                np.where(df[self.in_col].str.contains(row[self.keys]),
                         row[self.values], df[self.out_col]))
        if self.out_type == str:
            df[self.out_col].fillna('').astype(str)
        elif self.out_type == int:
            df[self.out_col] = pd.to_numeric(df[self.out_col],
                                             errors = 'coerce',
                                             downcast = 'integer')
        elif self.out_type == float:
            df[self.out_col] = pd.to_numeric(df[self.out_col],
                                             errors = 'coerce',
                                             downcast = float)
        return df

    def _str_series_match(self, df, in_string):
        for key, value in self.expressions.items():
            if re.search(key, in_string):
                df[self.out_col] = value
                break
            else:
                df[self.out_col] = self.default
        if self.out_type == str:
            df[self.out_col] = str(df[self.out_col])
        elif self.out_type == int:
            df[self.out_col] = int(df[self.out_col])
        elif self.out_type == float:
            df[self.out_col] = float(df[self.out_col])
        return df

    def _str_match(self, df, in_string, is_df):
        if is_df:
            df = self._str_df_match(df)
        else:
            df = self._str_series_match(df, in_string)
        return df

    def _pattern_df_match(self, df):
        for key, value in self.expressions.items():
            df[self.out_col] = df[self.in_col].str.find(key)
        return df

    def _pattern_series_match(self, df, in_string):
        for key, value in self.expressions.items():
            if re.search(key, in_string):
                df[self.out_col] = value
                break
        return df

    def _pattern_match(self, df, in_string, is_df):
        if is_df:
            df = self._pattern_df_match(df)
        else:
            df = self._pattern_series_match(df, in_string)
        return df

    def _patterns_df_match(self, df):
        for key, value in self.expressions.items():
            df[value] = df[self.in_col].str.findall(key)
        return df

    def _patterns_series_match(self, df, in_string):
        for key, value in self.expressions.items():
            if re.search(key, in_string):
                df[value] = re.search(key, in_string).group(0)
        return df

    def _patterns_match(self, df, in_string, is_df):
        if is_df:
            df = self._patterns_df_match(df)
        else:
            df = self._patterns_series_match(df, in_string)
        return df

    def _df_matches(self, df_row):
        temp_list = []
        for key, value in self.expressions.items():
            temp_list += re.findall(key, df_row[self.in_col])
        if temp_list:
            temp_list = list(unique_everseen(temp_list))
            df_row[self.out_col].extend(temp_list)
        return df_row

    def _list_df_match(self, df):
        df[self.out_col] = np.empty((len(df), 0)).tolist()
        df = df.apply(self._df_matches, axis = 'columns')
        return df

    def _list_series_match(self, df, in_string):
        temp_list = []
        for key, value in self.expressions.items():
            temp_list += re.findall(key, in_string)
        if temp_list:
            temp_list = list(unique_everseen(temp_list))
            if self.convert_lists:
                temp_list = self._list_to_string(temp_list)
            df[self.out_col] = temp_list
        else:
            df[self.out_col] = [self.default_value]
        return df

    def _list_match(self, df, in_string, is_df):
        if is_df:
            df = self._list_df_match(df)
        else:
            df = self._list_series_match(df, in_string)
        return df

    def match(self, df, in_string = '', in_col = '', in_prefix = '',
              in_suffix = '', out_col = '', out_prefix = '', out_suffix = '',
              default_value = ''):
        for var in self.col_vars:
            if locals()[var]:
                setattr(self, var, locals()[var])
            else:
                setattr(self, var, getattr(self, 'stored_' + var))
        is_df = isinstance(df, pd.Dataframe)
        if not in_string and not is_df:
            in_string = df[self.in_col]
        if is_df and self.out_col and not self.out_col in df.columns:
            df[self.out_col] = self.default_value
        matchers = {bool : self._bool_match,
                    str : self._str_match,
                    int : self._str_match,
                    float : self._str_match,
                    'pattern' : self._pattern_match,
                    'patterns' : self._patterns_match,
                    list : self._list_match}
        matcher = matchers(self.out_type)
        matcher(df, in_string, is_df)
        return df