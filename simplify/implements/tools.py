import re

from more_itertools import unique_everseen
import pandas as pd

def add_prefix(iterable, prefix):
    """Adds prefix to list, dict keys, pandas dataframe, or pandas series."""
    if isinstance(iterable, list):
        return [f'{prefix}_{value}' for value in iterable]
    elif isinstance(iterable, dict):
        return {f'{prefix}_{key}' : value for key, value in iterable.items()}
    elif isinstance(iterable, pd.Series):
        return iterable.add_prefix(prefix)
    elif isinstance(iterable, pd.DataFrame):
        return iterable.add_prefix(prefix)

def add_suffix(iterable, suffix):
    """Adds suffix to list, dict keys, pandas dataframe, or pandas series."""
    if isinstance(iterable, list):
        return [f'{value}_{suffix}' for value in iterable]
    elif isinstance(iterable, dict):
        return {f'{key}_{suffix}' : value for key, value in iterable.items()}
    elif isinstance(iterable, pd.Series):
        return iterable.add_suffix(suffix)
    elif isinstance(iterable, pd.DataFrame):
        return iterable.add_suffix(suffix)

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
        out_value = variable
    return out_value

def listify(variable):
    """Checks to see if the variable is stored in a list. If not, the variable
    is converted to a list or a list of 'none' is created if the variable is
    empty.
    """
    if not variable:
        return ['none']
    elif isinstance(variable, list):
        return variable
    else:
        return [variable]

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


def typify(variable):
    """Converts strings to list (if ', ' is present), int, float, or boolean
    datatypes based upon the content of the string. If no alternative datatype
    is found, the variable is returned in its original form.
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