
from dataclasses import dataclass
import re

import pandas as pd

from ...managers import Step


@dataclass
class HarvestStep(Step):
    """Parent class for preprocessing steps in the siMpLify package."""

    def __post_init__(self):
        if not self.options:
            self.options = self.default_options()
        super().__post_init__()
        return self

    def _list_to_string(self, variable):
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

    def _no_breaks(self, variable, in_column = None):
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

    def _no_double_space(self, variable, in_column = None):
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

    def _remove_excess(self, variable, excess, in_column = None):
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

    def _word_count(self, variable):
        """Returns word court for a string."""
        return len(variable.split(' ')) - 1

    def prepare(self):
        self.techniques = {}
        for key in self.options.keys():
            getattr(self, '_prepare_' + key)()
        return self

    def start(self, ingredients, almanac):
        for technique, algorithm in self.techniques.items():
            ingredients = algorithm.start(ingredients, almanac)
        return ingredients