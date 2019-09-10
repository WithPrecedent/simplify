
from dataclasses import dataclass
from datetime import timedelta

from numpy import datetime64
from pandas.api.types import CategoricalDtype

from simplify.core.base import SimpleType


@dataclass
class DataTypes(SimpleType):
    """Stores dictionaries related to datatypes used by siMpLify package."""

    def __post_init__(self):
        super().__post_init__()
        return self

    def _define(self):
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
class FileTypes(SimpleType):
    """Stores dictionaries related to file types used by siMpLify package."""

    def __post_init__(self):
        super().__post_init__()
        return self

    def _define(self):
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
class ReTypes(SimpleType):
    """Stores dictionaries related to specialized types used by the ReTool
    subpackage.
    """
    def __post_init__(self):
        super().__post_init__()
        return self

    def _define(self):
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