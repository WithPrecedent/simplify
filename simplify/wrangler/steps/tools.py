
from dataclasses import dataclass
from functools import wraps
from inspect import getfullargspec
import re
import time
from types import FunctionType

import pandas as pd


""" Decorators """



""" Mixins """

@dataclass
class SimpleIterables(object):

    @staticmethod
    def edit_prefix(iterable, prefix):
        """Adds prefix to list, dict keys, pandas dataframe, or pandas series.
        """
        if isinstance(iterable, list):
            return [f'{prefix}_{value}' for value in iterable]
        elif isinstance(iterable, dict):
            return (
                {f'{prefix}_{key}': value for key, value in iterable.items()})
        elif isinstance(iterable, pd.Series):
            return iterable.edit_prefix(prefix)
        elif isinstance(iterable, pd.DataFrame):
            return iterable.edit_prefix(prefix)

    @staticmethod
    def edit_suffix(iterable, suffix):
        """Adds suffix to list, dict keys, pandas dataframe, or pandas series.
        """
        if isinstance(iterable, list):
            return [f'{value}_{suffix}' for value in iterable]
        elif isinstance(iterable, dict):
            return (
                {f'{key}_{suffix}': value for key, value in iterable.items()})
        elif isinstance(iterable, pd.Series):
            return iterable.edit_suffix(suffix)
        elif isinstance(iterable, pd.DataFrame):
            return iterable.edit_suffix(suffix)

@dataclass
class SimpleParser(object):

    @staticmethod
    def list_to_string(variable):
        """Converts a list to a string with a comma and space separating each
        item. The conversion applies whether variable is a simple list or
        pandas series
        """
        if isinstance(variable, pd.Series):
            out_value = variable.apply(', '.join)
        elif isinstance(variable, list):
            out_value = ', '.join(variable)
        else:
            msg = 'Value must be a list or pandas series containing lists'
            raise TypeError(msg)
        return out_value

    @staticmethod
    def no_breaks(variable, in_column = None):
        """Removes line breaks and replaces them with single spaces. Also,
        removes hyphens at the end of a line and connects the surrounding text.
        Takes either string, pandas series, or pandas dataframe as input and
        returns the same.
        """
        if isinstance(variable, pd.DataFrame):
            variable[in_column].str.replace('[a-z]-\n', '')
            variable[in_column].str.replace('\n', ' ')
        elif isinstance(variable, pd.Series):
            variable.str.replace('[a-z]-\n', '')
            variable.str.replace('\n', ' ')
        else:
            variable = re.sub('[a-z]-\n', '', variable)
            variable = re.sub('\n', ' ', variable)
        return variable

    @staticmethod
    def no_double_space(variable, in_column = None):
        """Removes double spaces and replaces them with single spaces from a
        string. Takes either string, pandas series, or pandas dataframe as
        input and returns the same.
        """
        if isinstance(variable, pd.DataFrame):
            variable[in_column].str.replace('  +', ' ')
        elif isinstance(variable, pd.Series):
            variable.str.replace('  +', ' ')
        else:
            variable = variable.replace('  +', ' ')
        return variable

    @staticmethod
    def remove_excess(variable, excess, in_column = None):
        """Removes excess text included when parsing text into sections and
        strips text. Takes either string, pandas series, or pandas dataframe as
        input and returns the same.
        """
        if isinstance(variable, pd.DataFrame):
            variable[in_column].str.replace(excess, '')
            variable[in_column].str.strip()
        elif isinstance(variable, pd.Series):
            variable.str.replace(excess, '')
            variable.str.strip()
        else:
            variable = re.sub(excess, '', variable)
            variable = variable.strip()
        return variable

    @staticmethod
    def word_count(variable):
        """Returns word court for a string."""
        return len(variable.split(' ')) - 1


@dataclass
class SimpleUtilities(object):

    def _check_lengths(self, variable1, variable2):
        """Returns boolean value whether two list variables are of the same
        length. If a string is passed, it is converted to a 1 item list for
        comparison.

        Args:
            variable1: string or list.
            variable2: string or list.
        """
        return len(listify(variable1) == listify(variable2))

