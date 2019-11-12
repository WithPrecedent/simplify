"""
.. module:: types
:synopsis: data and file type base classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0

Contents:

    SimpleType: abstract base class that serves as the base class for data,
        file, and other types throughout the siMpLify package.
    DataTypes: subclass of SimpleType which stores proxy names of datatypes
        used by siMpLify which are connected to python, numpy, and pandas
        datatypes. Also, default values for each datatype are stored, which
        are primarily used in filling missing data.
    FileTypes: subclass of SimpleType which stores proxy names of file formats
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
                options: a dict with strings as keys corresponding to datatype
                    values.
                default_values: a dict with strings as keys matching those in
                    'options' and default values for those datatypes.

    With those basic dictionaries, a reverse dictionary ('reversed_options') is
    created and a variety of dunder methods are implemented to make using the
    type structures easier.

    """

    options: object = None

    def __post_init__(self) -> None:
        self.draft()
        self._create_reversed()
        return self

    """ Dunder Methods """

    def __getattr__(self, attribute):
        """Returns dict methods applied to options attribute if those methods
        are sought from the class instance.

        Args:
            attribute (str): attribute sought.

        Returns:
            dict method applied to 'options' or attribute, if attribute exists.

        Raises:
            AttributeError: if a dunder attribute is sought or attribute does
                not exist.

        """
        # Intecepts common dict methods and applies them to 'options'.
        if attribute in ['clear', 'items', 'pop', 'keys', 'values']:
            return getattr(self.options, attribute)
        else:
            try:
                return self.__dict__[attribute]
            except KeyError:
                error = attribute + ' not found in ' + self.__class__.__name__
                raise AttributeError(error)

    def __getitem__(self, item):
        """Returns item if item is in 'options' or 'reversed_options'.

        Args:
            item (str): key to be found.

        """
        try:
            return self.options[item]
        except KeyError:
            try:
                return self.reversed_options[item]
            except KeyError:
                error = item + ' is not in a recognized type.'
                raise KeyError(error)

    def __iter__(self):
        """Returns 'options.items()' to mirror dict functionality."""
        return self.options.items()

    def __setitem__(self, key, value):
        """Sets 'key' to 'value' in 'options' and reverse in 'reversed_options'.

        If the class has 'default_values', then:
            if 'value' matches a 'key' in reversed_options, the same default
            value is applied. Otherwise, None is used as the default_value.

        Args:
            key (str): key name to be set in 'options'.
            value (str or type): value to be set in 'options'.

        """
        self.options.update({key: value})
        self.reversed_options.update({value: key})
        try:
            self.default_values.update({key: self.reversed_options[value]})
        except KeyError:
            self.default_values.update({key: None})
        except AttributeError:
            pass
        return self

    """ Private Methods """

    def _create_reversed(self):
        """ Creates reversed dictionary of 'options' and stores it in
        'reversed_options'.

        """
        self.reversed_options = {
            value: key for key, value in self.options.items()}
        return self

    """ Public Methods """

    @abstractmethod
    def draft(self) -> None:
        """Required method that sets default values for a subclass."""
        pass
        return self

    """ Python Dictionary Compatibility Methods """

    def update(self, datatypes):
        """Adds values to 'options' and recreates reversed dict to mirror
        dict functionality.

        Args:
            datatypes (dict): a dictionary with keys of datatype names and
                values of datatypes.

        """
        self.options.update(datatypes)
        self._create_reversed()
        return self


@dataclass
class DataTypes(SimpleType):
    """Stores dictionaries related to datatypes used by siMpLify package.

    All datatypes use string proxies to allow for easy calling or related
    methods and consistent naming structure throughout the package.

    """
    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        """Sets default values related to datatypes."""
        # Sets string names of various datatypes available.
        self.options = {
            'boolean': bool,
            'float': float,
            'integer': int,
            'string': object,
            'categorical': CategoricalDtype,
            'list': list,
            'datetime': datetime64,
            'timedelta': timedelta}
        # Sets default values for missing data based upon datatype of column.
        self.default_values = {
            'boolean': False,
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
            names (str or list (str)): name(s) for keys in the 'datatypes' dict.
            python_types (type or list (type)): a python, numpy, pandas, or
                custom datatype or list of the same.
            datatypes (dict): a dictionary with keys of datatype names and
                values of datatypes.

        """
        try:
            self.options.update(datatypes)
        except TypeError:
            pass
        try:
            self.options.update(dict(zip(names, python_types)))
        except TypeError:
            pass
        self._create_reversed()
        return self

    def edit_default_values(self, default_values):
        """Updates 'default_values' dict.

        Args:
            default_values (dict): dictionary with keys of strings of datatypes
                values of default value for that datatype.

        """
        self.default_values.update(default_values)
        return self


@dataclass
class FileTypes(SimpleType):
    """Stores dictionaries related to file types used by siMpLify package."""

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        """Sets default values related to filetypes."""
        # Sets string names of various datatypes available.
        self.options = {
            'csv': '.csv',
            'excel': '.xlsx',
            'feather': '.ftr',
            'h5': '.hdf',
            'hdf': '.hdf',
            'pickle': '.pkl',
            'png': '.png',
            'text': '.txt',
            'txt': '.txt'}
        return self