"""
.. module:: retool
:synopsis: fast regular expression tools for pandas data containers
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""


"""
ReTool provides fast, easy-to-use tools for using regular expressions on
pandas Series and DataFrames.

The ReTool class is the controller class which assembles the proper 
classes and methods to address the specified goals based upon the
arguments passed.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import re

from simplify.core.base import SimpleClass


@dataclass
class ReTool(SimpleClass):
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
    particular use case. The normal use case where ReTool has efficiency gains
    is with a very large dataframe and a relatively small (< 500 rows)
    expressions dictionary.
    """
    technique: str
    keys: str = 'keys'
    values: str = 'values'
    sections: str = 'sections'
    datatypes: str = 'datatypes'
    flags: object = None
    zipped: object = None
    file_path: str = ''
    encoding: str = 'windows-1252'
    section_prefix: str = 'section'
    edit_prefixes: bool = True
    auto_finalize: bool = True

    def __post_init__(self):
        self.draft()
        if self.auto_finalize:
            self.finalize()
        return self

    def _aggregate_flags(self, row):
        """Creates regex flag sequence for compiling regexes."""
        self.flags = None
        for name, flag in self.flag_options.items():
            if name in row.index and row[name]:
                if not self.flags:
                    self.flags = flag
                else:
                    self.flags |= flag
        return self

    def _check_ingredients(self, ingredients, df, source):
        if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
            if ingredients is None:
                error = 'ReTool requires either df or ingredients'
                raise AttributeError(error)
            else:
                df = ingredients['default_df']
                if isinstance(source, str):
                    source = getattr(ingredients, source)
                else:
                    source = ingredients.source
        else:
            if source is None:
                error = 'If df passed, ReTool also requires source.'
                raise AttributeError(error)
        return df, source

    def _compile_expressions(self):
        """Compiles regular expressions (whether built or loaded)."""
        for i, row in self.expressions.iterrows():
            self._aggregate_flags(row)
            if self.flags:
                self.expressions.loc[i, self.keys] = (
                    re.compile(self.expressions.loc[i, self.keys],
                               flags = self.flags))
            else:
                self.expressions.loc[i, self.keys] = (
                    re.compile(self.expressions.loc[i, self.keys]))
        return self

    def _convert_to_dict(self):
        self.expressions = self.expressions[[self.keys, self.values]]
        self.expressions = (self.expressions.set_index(
                self.keys).to_dict()[self.values])
        return self

    def draft(self):
        # Sets str names for corresponding regex compiling flags.
        self.flag_options = {'ignorecase': re.IGNORECASE,
                             'dotall': re.DOTALL,
                             'locale': re.LOCALE,
                             'multiline': re.MULTILINE,
                             'verbose': re.VERBOSE,
                             'ascii': re.ASCII}
        # Sets options for matcher classes.
        self.options = {'organize': ReOrganize,
                        'parse': ReSearch,
                        'keyword': ReFrame}
        return self

    def _set_matcher(self):
        # Sets matcher options based upon data datatype.
        parameters = {'expressions': self.expressions,
                      'sections': self.sections,
                      'datatypes': self.datatypes,
                      'edit_prefixes': self.edit_prefixes,
                      'section_prefix': self.section_prefix}
        self.matcher = self.options[self.technique](**parameters)
        self.matcher.default_values = self.default_values
        return self

    def finalize(self):
        if self.file_path:
            tool = ReLoad(keys = self.keys,
                          values = self.values,
                          sections = self.sections,
                          datatypes = self.datatypes,
                          file_path = self.file_path,
                          auto_finalize = self.auto_finalize,
                          encoding = self.encoding,
                          section_prefix = self.section_prefix,
                          flag_options = self.flag_options)
        else:
            tool = ReBuild(keys = self.keys,
                           values = self.values,
                           sections = self.sections,
                           datatypes = self.datatypes,
                           flags = self.flags,
                           zipped = self.zipped,
                           auto_finalize = self.auto_finalize,
                           section_prefix = self.section_prefix,
                           flag_options = self.flag_options)
            self.keys = tool.keys
            self.values = tool.values
        self.sections = tool.sections
        self.datatypes = tool.datatypes
        self.expressions = tool.expressions
        self._compile_expressions()
        self._convert_to_dict()
        self._set_matcher()
        return self

    def produce(self, ingredients = None, df = None, source = None,
              remove_from_source = True):
        df, source = self._check_ingredients(ingredients = ingredients,
                                             df = df,
                                             source = source)
        if remove_from_source:
            df, source = self.matcher.produce(df = df, source = source)
        else:
            df = self.matcher.produce(df = df, source = source)
        return ingredients

    def update(self, key, value):
        self.expressions.update({key: value})
        return self


@dataclass
class ReBuild(object):

    keys: str = 'keys'
    values: str = 'values'
    sections: str = 'sections'
    datatypes: str = 'datatypes'
    flags: object = None
    zipped: object = None
    auto_finalize: bool = True
    section_prefix: str = 'section'
    flag_options: object = None

    def __post_init__(self):
        if self.auto_finalize:
            self.finalize()
        return self

    def finalize(self):
        """Builds regular expressions table."""
        self.expressions = pd.DataFrame(list(zip(self.keys, self.values)),
                                        columns = ['keys', 'values'])
        # Implements common column naming scheme regardless of source of data
        # for the expressions table.
        self.keys = 'keys'
        self.values = 'values'
        for flag in self.flag_options.keys():
            if self.flags and flag in self.flags:
                self.expressions[flag] = True
        return self


@dataclass
class ReLoad(object):

    keys: str = 'keys'
    values: str = 'values'
    sections: str = 'sections'
    datatypes: str = 'datatypes'
    file_path: str = ''
    auto_finalize: bool = True
    encoding: str = 'windows-1252'
    section_prefix: str = 'section'
    flag_options: object = None

    def __post_init__(self):
        if self.auto_finalize:
            self.finalize()
        return self

    def _explode_sections(self):
        if 'section' in self.expressions:
            self.expressions.explode(column = 'section')
        return self

    def finalize(self):
        """Loads data for expressions table from .csv file, converts keys to
        strings, and removes a common encording error character.
        """
        self.expressions = (pd.read_csv(self.file_path,
                                        index_col = False,
                                        encoding = self.encoding)
                              .astype(str)
                              .replace('Â', ''))
        self._explode_sections()
        if 'section' in self.expressions:
            self.sections = (
                self.expressions.set_index('values').to_dict()['section'])
        if 'datatypes' in self.expressions:
            self.datatypes = (
                self.expressions.set_index('section').to_dict()['datatype'])
        return self


@dataclass
class ReMatch(object):

    def __post_init__(self):
        return self

    def _set_out_column(self):
        if self.edit_prefixes:
            self.out_column = self.section_prefix + '_' + self.value
        else:
            self.out_column = self.value
        return self

    def _set_source(self):
        if self.section_prefix:
            self.source = self.section_prefix + '_' + self.value
        else:
            self.source = self.value
        return self

    def produce(self, df):
        for self.key, self.value in self.expressions.items():
            self._set_source()
            self.section = self.sections[self.value]
            self.datatype = self.datatypes[self.section]
            self._set_out_column()
            df = getattr(self, '_' + self.datatype)(df = df)
        return df


@dataclass
class ReFrame(ReMatch):
    """Stores and applies vectorized string and regular expression matching
    methods (when possible) to pandas dataframes.

    ReFrame allows keys and values to be formed from pandas series,
    python lists, or imported from a .csv file.

    Using the match method, the user can pass a dataframe containing source
    and output columns. Regular expressions are used as keys and may be
    compliled with or without any flag selected.

    If out_type is boolean, new dataframe columns are created with headers derived
    from the values in the dictionary. A boolean value is returned.

    if out_type is 'pattern', a single column is used or created with the name
    passed in 'out_column.' The return is the matched values from the
    regular expression expressions table.

    If out_type is string, integer or float, a single column is used or created
    with the header name passed in 'out_column.' The return is the matched
    value of the key in the expressions table.

    If out_type is list, a single column is used or created with the header
    name passed in 'out_column.' The return is all matched patterns based upon
    a regular expression stored in a python list within each dataframe column.
    """
    expressions: object
    sections: object
    datatypes: object
    edit_prefixes: bool = True
    section_prefix: str = 'section'

    def __post_init__(self):
        return self

    def _boolean(self, df):
        df[self.out_column] = np.where(
                df[self.source].str.contains(self.key, True,
                self.default_values[self.datatype]))
        return df

    def _divider(self, df):
        df = df.join(df[self.source].str.split(
                self.key, expand = True).edit_prefix(self.value))
        return df

    def _float(self, df):
        df = self._string(df)
        df[self.out_column] = pd.to_numeric(df[self.out_column],
                                            errors = 'coerce',
                                            downcast = float)
        return df

    def _integer(self, df):
        df = self._string(df)
        df[self.out_column] = pd.to_numeric(df[self.out_column],
                                            errors = 'coerce',
                                            downcast = 'integer')
        return df

    def _pattern(self, df):
        df[self.out_column] = df[self.source].str.find(self.key)
        return df

    def _patterns(self, df):
        df[self.value] = df[self.source].str.findall(self.key)
        return df

    def _remove(self, df):
        df[self.value] = df[self.source].replace(self.key, '')
        return df

    def _string(self, df):
        df[self.out_column] = np.where(
                df[self.source].str.contains(self.key),
                     self.value, self.default_values[self.datatype])
        return df


@dataclass
class ReSearch(ReMatch):

    expressions: object
    sections: object
    datatypes: object
    edit_prefixes: bool = True
    section_prefix: str = 'section'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _boolean(self, df):
        if re.search(self.key, self.source):
            df[self.out_column] = True
        else:
            df[self.out_column] = False
        return df

    def _float(self, df):
        df = self._string(df)
        df[self.out_column] = float(df[self.out_column])
        return df

    def _integer(self, df):
        df = self._string(df)
        df[self.out_column] = int(df[self.out_column])
        return df

    def _list(self, df):
        temp_list = []
        temp_list += re.findall(self.key, self.source)
        if temp_list:
            temp_list = list(self.deduplicate(temp_list))
            df[self.out_column] = temp_list
        else:
            df[self.out_column] = self.default_values[self.datatype]
        return df

    def _pattern(self, df):
        if re.search(self.key, self.source):
            df[self.out_column] = self.value
        return df

    def _patterns(self, df):
        if re.search(self.key, self.source):
            df[self.out_column] = re.search(
                    self.key, self.source).group(0).strip()
        return df

    def _string(self, df):
        if re.search(self.key, self.source):
            df[self.out_column] = self.value
        else:
            df[self.out_column] = self.default_value
        df[self.out_column] = str(df[self.out_column])
        return df


@dataclass
class ReOrganize(ReMatch):
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
    expressions: object
    sections: object
    datatypes: object
    edit_prefixes: bool = True
    section_prefix: str = 'section'

    def __post_init__(self):
        return self

    def produce(self, df, source):
        for self.key, self.value in self.expressions.items():
            self._set_out_column()
            if re.search(self.key, source):
                df[self.out_column] = re.search(
                        self.key, source).group(0).strip()
                source = re.sub(self.key, '', source)
            else:
                df[self.out_column] = self.default_values['string']
        return df, source
    
    
@dataclass
class ReTypes(SimpleType):
    """Stores dictionaries related to specialized types used by the ReTool
    subpackage.
    """
    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        """Sets default attributes related to ReTool datatypes."""
        # Sets string names for python and special datatypes.
        self.name_to_type = {'boolean': bool,
                             'float': float,
                             'integer': int,
                             'list': list,
                             'pattern': 'pattern',
                             'patterns': 'patterns',
                             'remove': 'remove',
                             'replace': 'replace',
                             'string': str}
        # Sets default values for missing data based upon datatype of column.
        self.default_values = {'boolean': False,
                               'float': 0.0,
                               'integer': 0,
                               'list': [],
                               'pattern': '',
                               'patterns': [],
                               'remove': '',
                               'replace': '',
                               'string': ''}
        return self
