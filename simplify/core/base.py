
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

from more_itertools import unique_everseen
import pandas as pd
from tensorflow.test import is_gpu_available


@dataclass
class SimpleMachine(ABC):
    """Absract base class for major classes in siMpLify package to support
    a common class structure and allow sharing of access methods.

    To use the class, a subclass must have the following methods:
        _outline: a private method which sets the default values for the
            subclass, and usually includes the self.options dictionary.
        prepare: a method which, after the user has set all options in the
            preferred manner, constructs the objects which can parse, modify,
            process, or analyze data.
        start: method which applies the prepared objects to passed data or
            other variables.

    To use the access dunder methods, the subclass should include self.options,
    a dictionary containing different strategies, algorithms, containers, or
    other objects.

    If the subclass includes boolean attributes of auto_prepare or auto_start,
    and those attributes are set to True, then the prepare and/or start methods
    are called when the class is instanced.
    """

    def __post_init__(self):
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Creates menu attribute if string passed to menu when subclass was
        # instanced. Injects attributes from menu settings to subclass.
        if hasattr(self, 'menu'):
            self._check_menu()
        # Calls _outline private method to set up class defaults.
        self._outline()
        # Runs attribute checks from list in self.checks (if it exists).
        self._run_checks()
        # Calls prepare method if it exists and auto_prepare is True.
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
        # Calls start method if it exists and auto_start is True.
        if hasattr(self, 'auto_start') and self.auto_start:
            self.start()
        return self

    def __call__(self, menu, *args, **kwargs):
        """When called as a function, a subclass will return the start method
        after running __post_init__. Any args and kwargs will only be passed
        to the start method.

        Parameters:
            menu: an instance of Menu or path where a menu configuration file
                is located must be passed when a subclass is called as a
                function.
        """
        self.menu = menu
        self.auto_prepare = True
        self.__post_init__()
        return self.start(*args, **kwargs)

    def __contains__(self, item):
        """Checks if item is in self.options; returns boolean."""
        if item in self.options:
            return True
        else:
            return False

    def __delitem__(self, item):
        """Deletes item if in self.options or, if an instance attribute, it
        is assigned a value of None."""
        if item in self.options:
            del self.options[item]
        elif hasattr(self, item):
            setattr(self, item, None)
        else:
            error = item + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
        return self

    def __getattr__(self, attr):
        """Returns dict methods applied to options attribute if those methods
        are sought from the class instance.

        Parameters:
            attr: attribute sought.
        """
        if attr in ['clear', 'items', 'pop', 'keys', 'update', 'values']:
            return getattr(self.options, attr)
        elif attr in self.__dict__:
            return self.__dict__[attr]
        elif attr.startswith('__') and attr.endswith('__'):
            raise AttributeError
        else:
            return None

    def __getitem__(self, item):
        """Returns item if item is in self.options or is an atttribute."""
        if item in self.options:
            return self.options[item]
        elif hasattr(self, item):
            return getattr(self, item)
        else:
            return None

    def __iter__(self):
        """Returns options.items() to mirror dict functionality."""
        return self.options.items()

    def __repr__(self):
        """Returns __str__."""
        return self.__str__()

    def __setitem__(self, item, value):
        """Adds item and value to options dictionary."""
        self.options[item] = value
        return self

    def __str__(self):
        """Returns lowercase name of class."""
        return self.__class__.__name__.lower()

    def _check_gpu(self):
        """If gpu status is not set, checks if the local machine has a GPU
        capable of supporting included machine learning algorithms."""
        if hasattr(self, 'gpu'):
            if self.gpu and self.verbose:
                print('Using GPU')
            elif self.verbose:
                print('Using CPU')
        elif is_gpu_available:
            self.gpu = True
            if self.verbose:
                print('Using GPU')
        else:
            self.gpu = False
            if self.verbose:
                print('Using CPU')
        return self

    def _check_inventory(self):
        """Adds an Inventory instance with default menu if one is not passed
        when subclass is instanced.
        """
        if not hasattr(self, 'inventory') or not self.inventory:
            # Inventory imported within function to avoid circular dependency.
            from simplify.core.inventory import Inventory
            self.inventory = Inventory(menu = self.menu)
        return self
#
#    def _check_lengths(self, variable1, variable2):
#        """Returns boolean value whether two list variables are of the same
#        length. If a string is passed, it is converted to a 1 item list for
#        comparison.
#
#        Parameters:
#            variable1: string or list.
#            variable2: string or list.
#        """
#        return len(self.listify(variable1) == self.listify(variable2))

    def _check_menu(self):
        """Loads menu from an .ini file if a string is passed to menu instead
        of a menu instance. Injects sections of menu to subclass instance
        using user settings stored in or default.
        """
        if isinstance(self.menu, str):
            # Menu imported within function to avoid circular dependency.
            from simplify.core.menu import Menu
            self.menu = Menu(file_path = self.menu)
        # Adds attributes to class from appropriate sections of the menu.
        sections = ['general']
        if hasattr(self, 'menu_sections') and self.menu_sections:
            if isinstance(self.menu_sections, str):
                sections.append(self.menu_sections)
            else:
                sections.extend(self.menu_sections)
        if (hasattr(self, 'name')
                and self.name in self.menu.configuration
                and not self.name in sections):
            sections.append(self.name)
        self.menu.inject(instance = self, sections = sections)
        return self

    @abstractmethod
    def _outline(self):
        pass
        return self

    def _run_checks(self):
        """Checks attributes from self.checks and initializes them if they do
        not exist by calling the appropriate method. Those methods should
        have the prefix _check_ followed by the string in self.checks.
        """
        if hasattr(self, 'checks') and self.checks:
            for check in self.checks:
                getattr(self, '_check_' + check)()
        return self

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def deduplicate(iterable):
        """Adds suffix to list, pandas dataframe, or pandas series."""
        if isinstance(iterable, list):
            return list(unique_everseen(iterable))
# Needs implementation for pandas
        elif isinstance(iterable, pd.Series):
            return iterable
        elif isinstance(iterable, pd.DataFrame):
            return iterable

    @staticmethod
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

    def load(self, name = None, file_path = None, folder = None,
             file_name = None, file_format = None):
        """Loads object from file into subclass attribute.

        Parameters:
            name: name of attribute for  file contents to be stored.
            file_path: a complete file path for the file to be loaded.
            folder: a path to the folder where the file should be loaded from
                (not used if file_path is passed).
            file_name: a string containing the name of the file to be loaded
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                Inventory.extensions.
        """
        setattr(self, name, self.inventory.load(file_path = file_path,
                                                folder = folder,
                                                file_name = file_name,
                                                file_format = file_format))
        return self

    @abstractmethod
    def prepare(self):
        pass
        return self

    def save(self, variable = None, file_path = None, folder = None,
             file_name = None, file_format = None):
        """Exports a variable or attribute to disc.

        Parameters:
            variable: a python object or a string corresponding to a subclass
                attribute.
            file_path: a complete file path for the file to be saved.
            folder: a path to the folder where the file should be saved (not
                used if file_path is passed).
            file_name: a string containing the name of the file to be saved
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                Inventory.extensions.
        """
        if isinstance(variable, str):
            variable = getattr(self, variable)
        self.inventory.save(variable = variable,
                            file_path = file_path,
                            folder = folder,
                            file_name = file_name,
                            file_format = file_format)
        return

    @abstractmethod
    def start(self, variable = None):
        pass
        return variable