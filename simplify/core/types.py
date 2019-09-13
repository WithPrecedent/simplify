
from dataclasses import dataclass
from datetime import timedelta

from numpy import datetime64
from pandas.api.types import CategoricalDtype

from simplify.core.base import SimpleType


@dataclass
class ReTypes(SimpleType):
    """Stores dictionaries related to specialized types used by the ReTool
    subpackage.
    """
    def __post_init__(self):
        super().__post_init__()
        return self

    def plan(self):
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