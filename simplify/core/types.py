
from dataclasses import dataclass
from datetime import timedelta

from numpy import datetime64
from pandas.api.types import CategoricalDtype


@dataclass
class SimpleTypes(object):
    """Parent class for setting dictionaries related to data and file types.
    """

    def __post_init__(self):
        if hasattr(self, '_set_defaults'):
            self._set_defaults()
            self._create_reversed()
        return self

    def __getitem__(self, item):
        """Returns item if item is in name_to_type or type_to_name."""
        if item in self.name_to_type:
            return self.name_to_type[item]
        elif item in self.type_to_name:
            return self.type_to_name[item]
        else:
            error = item + ' is not in a recognized type.'
            raise KeyError(error)

    def __iter__(self):
        """Returns name_to_type.items() to mirror dict functionality."""
        return self.name_to_type.items()

    def __setitem__(self, item, value):
        """Sets item to value in name_to_type and reverse in type_to_name.
        If the class has default_values, then:
        if value matches a key in type_to_name, the same default value is
        applied. Otherwise, None is used as the default_value.
        """
        self.name_to_type.update({item : value})
        self.type_to_name.update({value : item})
        if hasattr(self, 'default_values'):
            if value in self.type_to_name:
                self.default_values.update({item : self.type_to_name[value]})
            else:
                self.default_values.update({item : None})
        return self

    def _create_reversed(self):
        """ Creates reversed dictionary of self.dataname_to_type."""
        self.type_to_name = {
            value : key for key, value in self.name_to_type.items()}
        return self

    def keys(self):
        """Returns keys from name_to_type to mirror dict functionality."""
        return self.name_to_type.keys()

    def set_type(self, name, python_type, default_value = None):
        """Adds or replaces datatype with corresponding default value, if
        applicable.
        """
        self.type_to_name.update({name : python_type})
        self._create_reversed()
        if default_value and hasattr(self, 'default_values'):
            self.default_values.update({name : default_value})
        return self

    def values(self):
        """Returns values from name_to_type to mirror dict functionality."""
        return self.name_to_type.values()

@dataclass
class DataTypes(SimpleTypes):
    """Stores dictionaries related to datatypes used by siMpLify package."""

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        """Sets default values related to datatypes."""
        # Sets string names of various datatypes available.
        self.name_to_type = {'boolean' : bool,
                             'float' : float,
                             'integer' : int,
                             'string' : object,
                             'categorical' : CategoricalDtype,
                             'list' : list,
                             'datetime' : datetime64,
                             'timedelta' : timedelta}
        # Sets default values for missing data based upon datatype of column.
        self.default_values = {'boolean' : False,
                               'float' : 0.0,
                               'integer' : 0,
                               'string' : '',
                               'categorical' : '',
                               'list' : [],
                               'datetime' : 1/1/1900,
                               'timedelta' : 0}
        return self

@dataclass
class FileTypes(SimpleTypes):
    """Stores dictionaries related to file types used by siMpLify package."""

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        """Sets default values related to filetypes."""
        # Sets string names of various datatypes available.
        self.name_to_type = {'csv' : '.csv',
                             'excel' : '.xlsx',
                             'feather' : '.ftr',
                             'h5' : '.hdf',
                             'hdf' : '.hdf',
                             'pickle' : '.pkl',
                             'png' : '.png',
                             'text' : '.txt',
                             'txt' : '.txt'}
        return self

@dataclass
class ReTypes(SimpleTypes):
    """Stores dictionaries related to specialized types used by the ReTool
    subpackage.
    """
    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        """Sets default values related to ReTool datatypes."""
        # Sets string names for python and special datatypes.
        self.name_to_type = {'boolean' : bool,
                             'float' : float,
                             'integer' : int,
                             'list' : list,
                             'pattern' : 'pattern',
                             'patterns' : 'patterns',
                             'remove' : 'remove',
                             'replace' : 'replace',
                             'string' : str}
        # Sets default values for missing data based upon datatype of column.
        self.default_values = {'boolean' : False,
                               'float' : 0.0,
                               'integer' : 0,
                               'list' : [],
                               'pattern' : '',
                               'patterns' : [],
                               'remove' : '',
                               'replace' : '',
                               'string' : ''}
        return self

