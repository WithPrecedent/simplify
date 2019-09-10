from functools import wraps
from inspect import getfullargspec
import re
import time
from types import FunctionType

import pandas as pd


def add_prefix(iterable, prefix):
    """Adds prefix to list, dict keys, pandas dataframe, or pandas series.
    """
    if isinstance(iterable, list):
        return [f'{prefix}_{value}' for value in iterable]
    elif isinstance(iterable, dict):
        return (
            {f'{prefix}_{key}' : value for key, value in iterable.items()})
    elif isinstance(iterable, pd.Series):
        return iterable.add_prefix(prefix)
    elif isinstance(iterable, pd.DataFrame):
        return iterable.add_prefix(prefix)

def add_suffix(iterable, suffix):
    """Adds suffix to list, dict keys, pandas dataframe, or pandas series.
    """
    if isinstance(iterable, list):
        return [f'{value}_{suffix}' for value in iterable]
    elif isinstance(iterable, dict):
        return (
            {f'{key}_{suffix}' : value for key, value in iterable.items()})
    elif isinstance(iterable, pd.Series):
        return iterable.add_suffix(suffix)
    elif isinstance(iterable, pd.DataFrame):
        return iterable.add_suffix(suffix)

def check_arguments(method, excludes = None):
    """Decorator which uses class instance attribute of the same name as a
    passed parameter if no argument is passed for that parameter and the
    parameter is not listed in excludes.

    Parameters:
        method: wrapped method within a class instance.
        excludes: list or string of parameters for which a local attribute
            should not be used.
    """
    if not excludes:
        excludes = []
    else:
        excludes = listify(excludes)

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        argspec = getfullargspec(method)
        unpassed_args = argspec.args[len(args):]
        for unpassed in unpassed_args:
            if unpassed not in excludes and hasattr(self, unpassed):
                kwargs.update({unpassed : getattr(self, unpassed)})
        return method(self, *args, **kwargs)

    return wrapper

def _check_lengths(variable1, variable2):
    """Returns boolean value whether two list variables are of the same
    length. If a string is passed, it is converted to a 1 item list for
    comparison.

    Parameters:
        variable1: string or list.
        variable2: string or list.
    """
    return len(self.listify(variable1) == self.listify(variable2))
        
def convert_time(seconds):
    """Function that converts seconds into hours, minutes, and seconds."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

def listify(variable):
    """Checks to see if the variable is stored in a list. If not, the
    variable is converted to a list or a list of 'none' is created if the
    variable is empty.
    """
    if not variable:
        return ['none']
    elif isinstance(variable, list):
        return variable
    else:
        return [variable]

def deduplicate(iterable):
    """Adds suffix to list, pandas dataframe, or pandas series."""
    if isinstance(iterable, list):
        return list(unique_everseen(iterable))
# Needs implementation for pandas
    elif isinstance(iterable, pd.Series):
        return iterable
    elif isinstance(iterable, pd.DataFrame):
        return iterable

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

def timer(process = None):
    """Decorator for computing the length of time a process takes.

    Parameters:
        process: string containing name of class or method."""

    if not process:
        if isinstance(process, FunctionType):
            process = process.__name__
        else:
            process = process.__class__.__name__

    def shell_timer(_function):

        def decorated(*args, **kwargs):
            start_time = time.time()
            result = _function(*args, **kwargs)
            total_time = time.time() - start_time
            h, m, s = convert_time(total_time)
            print(f'{process} completed in %d:%02d:%02d' % (h, m, s))
            return result

        return decorated

    return shell_timer

def typify(variable):
    """Converts strings to list (if ', ' is present), int, float, or
    boolean datatypes based upon the content of the string. If no
    alternative datatype is found, the variable is returned in its original
    form.
    """
    if ', ' in variable:
        return variable.split(', ')
    elif re.search('\d', variable):
        try:
            return int(variable)
        except ValueError:
            try:
                return float(variable)
            except ValueError:
                return variable
    elif variable in ['True', 'true', 'TRUE']:
        return True
    elif variable in ['False', 'false', 'FALSE']:
        return False
    elif variable in ['None', 'none', 'NONE']:
        return None
    else:
        return variable

def word_count(variable):
    """Returns word court for a string."""
    return len(variable.split(' ')) - 1