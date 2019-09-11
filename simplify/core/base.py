
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import warnings

from more_itertools import unique_everseen
import pandas as pd
from tensorflow.test import is_gpu_available


@dataclass
class SimpleClass(ABC):
    """Absract base class for major classes in siMpLify package to support
    a common class structure and allow sharing of access methods.

    To use the class, a subclass must have the following methods:
        _define: a private method which sets the default values for the
            subclass, and usually includes the self.options dictionary.
        prepare: a method which, after the user has set all options in the
            preferred manner, constructs the objects which can parse, modify,
            process, or analyze data.
    The following methods are not strictly required but should be used if
    the subclass is transforming data or other variable (as opposed to merely
    containing data or variables):
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
        if hasattr(self, 'menu') and self.__class__.__name__ != 'Menu':
            self._check_menu()
        # Calls _define private method to set up class defaults.
        self._define()
        # Runs attribute checks from list in self.checks (if it exists).
        self._run_checks()
        # Calls prepare method if it exists and auto_prepare is True.
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
            # Calls start method if it exists and auto_start is True.
            if hasattr(self, 'auto_start') and self.auto_start:
                self.start()
        return self

    """ Magic Methods """

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
        self.auto_start = False
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

    def __setitem__(self, item, value):
        """Adds item and value to options dictionary."""
        self.options[item] = value
        return self

    """ Private Methods """

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

    def _check_ingredients(self, ingredients = None):
        """Checks if ingredients attribute exists. If so, it determines if it
        contains a file folder, file path, or Ingredients instance. Depending
        upon its type, different actions are taken to actually create an
        Ingredients instance. If ingredients is None, then an Ingredients
        instance is created with no pandas DataFrames within it.

        Parameters:
            ingredients: an Ingredients instance, a file path containing a
                DataFrame or Series to add to an Ingredients instance, or
                a folder containing files to be used to compose Ingredients
                DataFrames and/or Series.
        """
        # Ingredients imported within function to avoid circular dependency.
        from simplify.core.ingredients import Ingredients
        if ingredients:
            self.ingredients = ingredients
        if (isinstance(self.ingredients, pd.Series)
                or isinstance(self.ingredients, pd.DataFrame)):
            self.ingredients = Ingredients(df = self.ingredients)
        elif isinstance(self.ingredients, str):
            if os.path.isfile(self.ingredients):
                df = self.inventory.load(folder = self.inventory.data,
                                         file_name = self.ingredients)
                self.ingredients = Ingredients(df = df)
            elif os.path.isdir(self.ingredients):
                self.inventory.create_glob(folder = self.ingredients)
                self.ingredients = Ingredients()
        elif not self.ingredients:
            self.ingredients = Ingredients()
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

    def _check_steps(self):
        if not hasattr(self, 'steps') or not self.steps:
            if hasattr(self, self.name + '_steps'):
                self.steps = self.listify(getattr(self, self.name + '_steps'))
            else:
                self.steps = []
        else:
            self.steps = self.listify(self.steps)
        if not hasattr(self, 'step') or not self.step:
            self.step = self.steps[0]
        return self

    @abstractmethod
    def _define(self):
        """Required method that sets default values for a subclass.

        A dict called 'options' should be defined here for subclasses to use
        much of the functionality of SimpleClass.

        Generally, the 'checks' attribute should be set here if the subclass
        wants to make use of related methods.
        """
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

    """ Public Methods """

    def conform(self, step):
        """Sets 'step' attribute to current step in siMpLify. This is used
        to maintain a universal state in the package for subclasses that are
        state dependent.
        """
        self.step = step
        return self

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
        """Required method which creates any objects to be applied to data or
        variables. In the case of iterative classes, such as Cookbook, this
        method should construct any plans to be later implemented by the start
        method.
        """
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

    def start(self, variable = None, **kwargs):
        """Optional method that implements all of the prepared objects on the
        passed variable. The variable is returned after being transformed by
        called methods.

        Parameters:
            variable: any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        pass
        return variable

@dataclass
class SimpleType(ABC):
    """Parent abstract base class for setting dictionaries related to data and
    file types.

    To use the class, a subclass must have the following methods:
        _define: a private method which sets the default values for the
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
        if hasattr(self, '_define'):
            self._define()
            self._create_reversed()
        return self

    """ Magic Methods """

    def __getattr__(self, attr):
        """Returns dict methods applied to 'name_to_type' attribute if those
        methods are sought from the class instance.

        Parameters:
            attr: attribute sought.
        """
        if attr in ['clear', 'items', 'pop', 'keys', 'update', 'values']:
            return getattr(self.name_to_type, attr)
        elif attr in self.__dict__:
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

    def __setitem__(self, item, value):
        """Sets item to 'value' in 'name_to_type' and reverse in 'type_to_name'.
        If the class has 'default_values', then:
            if 'value' matches a 'key' in type_to_name, the same default value
            is applied. Otherwise, None is used as the default_value.
        """
        self.name_to_type.update({item : value})
        self.type_to_name.update({value : item})
        if hasattr(self, 'default_values'):
            if value in self.type_to_name:
                self.default_values.update({item : self.type_to_name[value]})
            else:
                self.default_values.update({item : None})
        return self

    """ Private Methods """

    def _create_reversed(self):
        """ Creates reversed dictionary of 'name_to_type' and stores it in
        'type_to_name'.
        """
        self.type_to_name = {
            value : key for key, value in self.name_to_type.items()}
        return self

    @abstractmethod
    def _define(self):
        """Required method that sets default values for a subclass."""
        pass
        return self

    """ Public Methods """

    def keys(self):
        """Returns keys from 'name_to_type' to mirror dict functionality."""
        return self.name_to_type.keys()

    def save(self):
        """Exports the subclass to disc in pickle format."""
        export_folder = getattr(self.inventory, self.export_folder)
        file_name = self.__class__.__name__.lower()
        self.inventory.save(variable = self,
                            folder = export_folder,
                            file_name = file_name,
                            file_format = 'pickle')
        return self

    def update(self, name, python_type, default_value = None):
        """Adds or replaces datatype with corresponding default value, if
        applicable.
        """
        self.type_to_name.update({name : python_type})
        self._create_reversed()
        if default_value and hasattr(self, 'default_values'):
            self.default_values.update({name : default_value})
        return self

    def values(self):
        """Returns values from 'name_to_type' to mirror dict functionality."""
        return self.name_to_type.values()