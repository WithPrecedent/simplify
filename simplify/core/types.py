"""


Contents:

    SimpleType: abstract base class that serves as the parent class for data,
        file, and other types throughout the siMpLify package.
    DataType: subclass of SimpleType which stores proxy names of datatypes
        used by siMpLify which are connected to python, numpy, and pandas
        datatypes. Also, default values for each datatype are stored, which
        are primarily used in filling missing data.
    FileType: subclass of SimpleType which stores proxy names of file formats
        linked to file extensions used for loading and saving files.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta

from numpy import datetime64
from pandas.api.types import CategoricalDtype
   
@dataclass
class SimpleType(ABC):
    """Parent abstract base class for setting dictionaries related to datatypes 
    and file types.

    To use the class, a subclass must have the following methods:
        draft: a method which sets the default values for the
            subclass. This method should define two dictionaries:
                name_to_type: a dict with strings as keys corresponding to
                    datatype values.
                default_values: a dict with strings as keys matching those in
                    'name_to_type' and default values for those datatypes.

    With those basic dictionaries, a reverse dictionary ('type_to_name') is
    created and a variety of dunder methods are implemented to make using the
    type structures easier.
    """

    def __post_init__(self):
        self.draft()
        self._create_reversed()
        return self

    """ Magic Methods """

    def __getattr__(self, attr):
        """Returns dict methods applied to 'name_to_type' attribute if those
        methods are sought from the class instance. 

        Args:
            attr: attribute sought.
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif attr.startswith('__') and attr.endswith('__'):
            raise AttributeError
        else:
            return None

    def __getitem__(self, item):
        """Returns item if item is in 'name_to_type' or 'type_to_name'."""
        if item in self.name_to_type:
            return self.name_to_type[item]
        elif item in self.type_to_name:
            return self.type_to_name[item]
        else:
            error = item + ' is not in a recognized type.'
            raise KeyError(error)

    def __iter__(self):
        """Returns 'name_to_type.items()' to mirror dict functionality."""
        return self.name_to_type.items()

    def __setitem__(self, key, value):
        """Sets key to 'value' in 'name_to_type' and reverse in 'type_to_name'.
        
        If the class has 'default_values', then:
            if 'value' matches a 'key' in type_to_name, the same default value
            is applied. Otherwise, None is used as the default_value.
            
        Args:
            item: key name to be set in 'name_to_type'.
            value: value to be set in 'name_to_type'.
        """
        self.name_to_type.update({key: value})
        self.type_to_name.update({value: key})
        if hasattr(self, 'default_values'):
            if value in self.type_to_name:
                self.default_values.update({key: self.type_to_name[value]})
            else:
                self.default_values.update({key: None})
        return self

    """ Private Methods """

    def _create_reversed(self):
        """ Creates reversed dictionary of 'name_to_type' and stores it in
        'type_to_name'.
        """
        self.type_to_name = {
            value: key for key, value in self.name_to_type.items()}
        return self

    """ Public Methods """

    @abstractmethod
    def draft(self):
        """Required method that sets default values for a subclass."""
        pass
        return self
  
    def items(self):
        """Returns items from 'name_to_type' to mirror dict functionality."""
        return self.name_to_type.items()   
            
    def keys(self):
        """Returns keys from 'name_to_type' to mirror dict functionality."""
        return self.name_to_type.keys()
    
    def pop(self, key):
        """Removes key from 'name_to_type' and 'type_to_name' to mirror dict 
        functionality.
        
        Args:
            key: dict key to be removed.
        """
        self.name_to_type.pop(key)
        self.type_to_name.pop(key)
        return self
    
    def update(self, datatypes):
        """Adds values to 'name_to_type' and recreates reversed dict to mirror
        dict functionality.
        
        Args:
            datatypes: a dictionary with keys of datatype names and values of 
                datatypes.        
        """
        self.name_to_type.update(datatypes)
        self._create_reversed()
        return self

    def values(self):
        """Returns values from 'name_to_type' to mirror dict functionality."""
        return self.name_to_type.values()


@dataclass
class DataTypes(SimpleType):
    """Stores dictionaries related to datatypes used by siMpLify package.
    
    All datatypes use string proxies to allow for easy calling or related 
    methods and consistent naming structure throughout the package.
    """

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        """Sets default values related to datatypes."""
        # Sets string names of various datatypes available.
        self.name_to_type = {'boolean': bool,
                             'float': float,
                             'integer': int,
                             'string': object,
                             'categorical': CategoricalDtype,
                             'list': list,
                             'datetime': datetime64,
                             'timedelta': timedelta}
        # Sets default values for missing data based upon datatype of column.
        self.default_values = {'boolean': False,
                               'float': 0.0,
                               'integer': 0,
                               'string': '',
                               'categorical': '',
                               'list': [],
                               'datetime': 1/1/1900,
                               'timedelta': 0}
        return self
     
    def edit_datatypes(self, names = None, python_types = None, 
                       datatypes = None):
        """Updates datatypes dictionary and its reverse with new keys and values
        from passed arguments.
        
        Args:
            names: a string name or list of names for keys in the datatypes
                dict.
            python_types: a python, numpy, pandas, or custom datatype or list
                of the same.
            datatypes: a dictionary with keys of datatype names and values of 
                datatypes.
        """
        if datatypes:
            self.name_to_type.update(datatypes)
        if names and python_types:
            self.name_to_type.update(dict(zip(names, python_types)))
        self._create_reversed()
        return self  
       
    def edit_default_values(self, default_values):
        """Updates 'default_values' dict'
        
        Args:
            default_values: dict with keys of strings of datatypes and values
                of default value for that datatype.
        """
        self.default_values.update(default_values)
        return self

  
@dataclass
class FileTypes(SimpleType):
    """Stores dictionaries related to file types used by siMpLify package."""

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        """Sets default values related to filetypes."""
        # Sets string names of various datatypes available.
        self.name_to_type = {'csv': '.csv',
                             'excel': '.xlsx',
                             'feather': '.ftr',
                             'h5': '.hdf',
                             'hdf': '.hdf',
                             'pickle': '.pkl',
                             'png': '.png',
                             'text': '.txt',
                             'txt': '.txt'}
        return self