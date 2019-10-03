"""
.. module:: base
:synopsis: core parent classes of siMpLify package
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
import os
import warnings

from more_itertools import unique_everseen
import pandas as pd


@dataclass
class SimpleClass(ABC):
    """Absract base class for classes in siMpLify package to support a common
    architecture and allow for sharing of universal methods.

    SimpleClass creates a code structure patterned after the writing process.
    It divides processes into four stages which are the names or prefixes to
    the core methods used throughout the siMpLify package:

        1) draft: sets default attributes (required).
        2) edit: makes any desired changes to the default attributes.
        3) publish: creates python objects based upon those attributes.
        4) read: applies those publishd objects to passed variables
            (usually data).

    If the subclass includes boolean attributes of 'auto_publish' or
    'auto_read', and those attributes are set to True, then the 'publish'
    and/or 'read' methods are called when the class is instanced.

    Args:
        These arguments are not required for any SimpleClass, but are commonly
        used throughout the package. Brief descriptions are included here:

        idea(Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is created by a subclass of SimpleClass, it is
            automatically made available to all other SimpleClass subclasses
            that are instanced in the future.
        depot(Depot): an instance of Depot. Once a Depot instance is created by
            a subclass of SimpleClass, it is automatically made available to all
            other SimpleClass subclasses that are instanced in the future.
        ingredients(Ingredients or str): an instance of Ingredients of a string
            containing the full file path of where a supported file type that
            can be loaded into a pandas DataFrame is located. If it is a string,
            the loaded DataFrame will be bound to a new ingredients instance as
            the 'df' attribute.
        name(str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.
        auto_read(bool): whether to call the 'read' method when the class
            is instanced.

    """

    def __post_init__(self):
        """Calls selected initialization methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Creates 'idea' attribute if a string is passed to Idea when subclass
        # was instanced. Injects attributes from 'idea' to subclass.
        if self.__class__.__name__ != 'Idea':
            self._check_idea()

        # Calls draft method to set up class instance defaults.
        self.draft()
        # Runs attribute checks from list in 'checks' attribute (if it exists).
        self._run_checks()
        # Converts values in 'options' to classes by lazily importing them.
        self._lazily_import_options()
        # Registers subclass into lists based upon specific subclass needs.
        self._register_subclass()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
            # Calls 'read' method if 'auto_read' is True.
            if hasattr(self, 'auto_read') and self.auto_read:
                self.read()
        return self

    """ Magic Methods """

    def __call__(self, idea, *args, **kwargs):
        """When called as a function, a subclass will return the read method
        after running __post_init__.

        Args:
            idea(Idea or str): an instance of Idea or path where an Idea
                configuration file is located. This argument must be passed when
                a subclass is called as a function.
            *args and **kwargs (any): passed to the 'read' method.

        Returns:
            return value of 'read' method.

        """
        self.idea = idea
        self.auto_publish = True
        self.auto_read = False
        self.__post_init__()
        return self.read(*args, **kwargs)

    def __contains__(self, item):
        """Checks if item is in 'options'.

        Args:
            item(str): item to be searched for in 'options' keys.

        Returns:
            True, if 'item' in 'options' - otherwise False.

        """
        return item in self.options

    def __delitem__(self, item):
        """Deletes item if in 'options' or, if it is an instance attribute, it
        is assigned a value of None.

        Args:
            item(str): item to be deleted from 'options' or attribute to be
                assigned None.

        Raises:
            KeyError: if item is neither in 'options' or an attribute.

        """
        if item in self.options:
            del self.options[item]
        elif self.exists(item):
            setattr(self, item, None)
        else:
            error = item + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
        return self

    def __getattr__(self, attr):
        """Returns dict methods applied to options attribute if those methods
        are sought from the class instance.

        Args:
            attr(str): attribute sought.

        Returns:
            dict method applied to 'options' or attribute, if attribute exists.

        Raises:
            AttributeError: if a dunder attribute is sought or attribute does
                not exist.

        """
        # Intercepts common dict methods and applies them to 'options' dict.
        if attr in ['clear', 'items', 'pop', 'keys', 'update', 'values']:
            return getattr(self.options, attr)
        elif attr in self.__dict__:
            return self.__dict__[attr]
        elif attr.startswith('__') and attr.endswith('__'):
            error = 'Access to magic methods not permitted through __getattr__'
            raise AttributeError(error)
        else:
            error = attr + ' not found in ' + self.__class__.__name__
            raise AttributeError(error)

    def __getitem__(self, item):
        """Returns item if 'item' is in 'options' or is an atttribute.

        Args:
            item(str): item matching 'options' dictionary key or attribute name.

        Returns:
            Value for item in 'options', 'item' attribute value, or None if
                neither of those exist.

        """
        if item in self.options:
            return self.options[item]
        elif hasattr(self, item):
            return getattr(self, item)
        else:
            return None

    def __iter__(self):
        """Returns 'options' to mirror dictionary functionality."""
        return self.options

    def __setitem__(self, item, value):
        """Adds item and value to options dictionary.

        Args:
            item(str): 'options' key to be set.
            value(any): corresponding value to be set for 'item' key in
                'options'.

        """
        self.options[item] = value
        return self

    """ Private Methods """

    def _check_depot(self):
        """Adds a Depot instance with default settings as 'depot' attribute if
        one was not passed when the subclass was instanced.
        """
        # Local import to avoid circular dependency.
        from simplify import Depot
        if self.exists('depot'):
            if isinstance(self.depot, str):
                self.depot = Depot(root_folder = self.depot)
        else:
            self.depot = Depot()
        return self

    def _check_gpu(self):
        """If gpu status is not set, checks if the local machine has a GPU
        capable of supporting included machine learning algorithms.

        Because the tensorflow 'is_gpu_available' method is very lenient in
        counting what qualifies, it is recommended to set the 'gpu' attribute
        directly or through an Idea instance.
        """
#        if hasattr(self, 'gpu'):
#            if self.gpu and self.verbose:
#                print('Using GPU')
#            elif self.verbose:
#                print('Using CPU')
#        elif is_gpu_available:
#            self.gpu = True
#            if self.verbose:
#                print('Using GPU')
#        else:
#            self.gpu = False
#            if self.verbose:
#                print('Using CPU')
        return self

    def _check_idea(self):
        """Loads Idea settings from a file if a string is passed instead of an
        Idea instance.

        Injects sections of 'idea' to a subclass instance using
        user settings.
        """
        # Local import to avoid circular dependency.
        from simplify import Idea
        if self.exists('idea') and isinstance(self.idea, str):
            self.idea = Idea(configuration = self.idea)
        # Adds attributes to class from appropriate sections of the idea.
        sections = ['general']
        if self.exists('idea_sections'):
            if isinstance(self.idea_sections, str):
                sections.append(self.idea_sections)
            elif isinstance(self.idea_sections, list):
                sections.extend(self.idea_sections)
        if (hasattr(self, 'name')
                and self.name in self.idea.configuration
                and not self.name in sections):
            sections.append(self.name)
        self = self.idea.inject(instance = self, sections = sections)
        return self

    def _check_ingredients(self, ingredients = None):
        """Checks if ingredients attribute exists and takes appropriate action.

        If an 'ingredients' attribute exists, it determines if it contains a
        file folder, file path, or Ingredients instance. Depending upon its
        type, different actions are taken to actually create an Ingredients
        instance.

        If ingredients is None, then an Ingredients instance is
        created with no pandas DataFrames or Series within it.

        Args:
            ingredients (Ingredients, a file path containing a DataFrame or
                Series to add to an Ingredients instance, or a folder
                containing files to be used to compose Ingredients DataFrames
                and/or Series).
        """
        # Local import to avoid circular dependency.
        from simplify import Ingredients
        if ingredients:
            self.ingredients = ingredients
        if (isinstance(self.ingredients, pd.Series)
                or isinstance(self.ingredients, pd.DataFrame)):
            self.ingredients = Ingredients(df = self.ingredients)
        elif isinstance(self.ingredients, str):
            if os.path.isfile(self.ingredients):
                df = self.depot.load(folder = self.depot.data,
                                     file_name = self.ingredients)
                self.ingredients = Ingredients(df = df)
            elif os.path.isdir(self.ingredients):
                self.depot.create_glob(folder = self.ingredients)
                self.ingredients = Ingredients()
        elif not self.ingredients:
            self.ingredients = Ingredients()
        return self

    def _check_steps(self):
        """Creates 'steps' attribute from one of several sources.

        Initially, the method checks for 'steps' attribute. If that doesn't
        exist, it searches for an attribute with the class 'name' prefix
        followed by 'steps' (e.g. 'cookbook_steps'). This latter attribute is
        often created from the Idea instance.

        If the value stored in the 'steps' attribute is 'all', 'none', or
        'default', the steps value is changed using the '_convert_wildcards'
        method.

        Also, a 'step' attribute is created from the first item in the 'steps'
        attribute.
        """
        if not self.exists('steps'):
            if hasattr(self, self.name + '_steps'):
                self.steps = self._convert_wildcards(getattr(
                        self, self.name + '_steps'))
            elif self.exists('options'):
                self.steps = list(self.options.keys())
        else:
            self.steps = self.listify(self.steps)
        if not self.exists('step'):
            self.step = self.steps[0]
        return self

    def _convert_wildcards(self, value):
        """Converts 'all', 'default', or 'none' values to a list of items.

        Args:
            value (list or str): name(s) of techniques, steps, or managers.

        Returns:
            If 'all', all keys listed in 'options' dictionary are returned.
            If 'default', 'default_operations' are returned or, if they don't
                exist, all keys listed in 'options' dictionary are returned.
            Otherwise, 'techniques' is returned intact.
        """
        if value in ['all', ['all']]:
            return list(self.options.keys())
        elif value in ['default', ['default']]:
            if (hasattr(self, 'default_operations')
                    and self.default_operations):
                return self.default_operations
            else:
                return list(self.options.keys())
        elif value in ['none', ['none'], None]:
            return 'none'
        else:
            return value

    def _lazily_import_options(self):
        """Limits module imports to only needed package dependencies.

        This method allows users to either save memory or have less
        dependencies locally available by importing fewer packages than
        would be done through normal, blanket importation.

        To use this method, 'options' should be formatted as follows:
            {name(str): [module_path(str), class_name(str)]}

        """
        imported_options = {}
        if self.exists('options'):
            if not hasattr(self, 'lazy_import') or self.lazy_import:
                if self.has_list_values(self.options):
                    for name, settings in self.options.items():
                        imported_options.update(
                            {name: getattr(import_module(settings[0]),
                                           settings[1])})
                    self.options = imported_options
        return self

    def _register_subclass(self):
        """Adds subclass to appropriate list based on attribute in subclass."""
        self._registered_subclasses = {
            'registered_state_subclasses': 'state_dependent'}
        for subclass_list, attribute in self._registered_subclasses.items():
            if not hasattr(self, subclass_list):
                setattr(self, subclass_list, [])
            if hasattr(self, attribute) and getattr(self, attribute):
                getattr(self, subclass_list).append(self)
        return self

    def _run_checks(self):
        """Checks attributes from 'checks' and runs corresponding methods based
        upon strings stored in 'checks'.

        Those methods should have the prefix '_check_' followed by the string
        in the attribute 'checks' and have no parameters other than 'self'. Any
        subclass seeking to add new checks can add a new method using those
        naming conventions.
        """
        if self.exists('checks'):
            for check in self.checks:
                getattr(self, '_check_' + check)()
        return self

    """ Public Tool Methods """

    def conform(self, step):
        """Sets 'step' attribute to passed 'step' throughout package.

        This method is used to maintain a universal state in the package for
        subclasses that are state dependent. It iterates through any subclasses
        listed in 'registered_state_subclasses' to call their 'conform'
        methods.

        Args:
            step(str): corresponds to current state in siMpLify package.
        """
        self.step = step
        for _subclass in self.registered_state_subclasses:
            _subclass.conform(step = step)
        return self

    @staticmethod
    def deduplicate(iterable):
        """Deduplicates list, pandas DataFrame, or pandas Series.

        Args:
            iterable (list, DataFrame, or Series): iterable to have duplicate
                entries removed.

        Returns:
            iterable (list, DataFrame, or Series, same as passed type):
                iterable with duplicate entries removed.
        """
        if isinstance(iterable, list):
            return list(unique_everseen(iterable))
        elif isinstance(iterable, pd.Series):
            return iterable.drop_duplicates(inplace = True)
        elif isinstance(iterable, pd.DataFrame):
            return iterable.drop_duplicates(inplace = True)

    @staticmethod
    def dictify(keys, values, ignore_values_list = False):
        """Creates dict from list of keys and same value or zips two lists.

        Args:
            keys (list): keys for new dict.
            values (any): valuse for all keys in the new dict or list of values
                corresponding to list of keys.
            ignore_values_list (bool): if value is a list, but the list should
                be the value for all keys, set to True.

        Returns:
            dict with 'keys' as keys and 'values' as all values or zips two
            lists together to form a dict.
        """
        if isinstance(values, list) and ignore_values_list:
            return dict.fromkeys(keys, values)
        else:
            return dict(zip(keys, values))

    def exists(self, attribute):
        """Returns if attribute exists in subclass and is not None.

        Args:
            attribute (str): name of attribute to be evaluated.

        Returns:
            boolean value indicating whether the attribute exists and is not
                None.
        """
        return (hasattr(self, attribute)
                and getattr(self, attribute) is not None)

    @staticmethod
    def has_list_values(dictionary):
        """Returns if passed 'dictionary' has lists for values.

        Args:
            dictionary (dict): dict to be tested.

        Returns:
            boolean value indicating whether any value in the 'dictionary' has
                list for all values
        """
        return all(isinstance(d, list) for d in dictionary.values())

    @staticmethod
    def is_nested(dictionary):
        """Returns if passed 'dictionary' is nested at least one-level.

        Args:
            dictionary (dict): dict to be tested.

        Returns:
            boolean value indicating whether any value in the 'dictionary' is
                also a dict (meaning that 'dictionary' is nested).
        """
        return any(isinstance(d, dict) for d in dictionary.values())

    @staticmethod
    def listify(variable):
        """Stores passed variable as a list (if not already a list).

        Args:
            variable (str or list): variable to be transformed into a list to
                allow iteration.

        Returns:
            variable (list): either the original list, a string converted to a
                list, or a list containing 'none' as its only item.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

    @staticmethod
    def stringify(variable):
        """Converts one item list to a string (if not already a string).

        Args:
            variable (list): variable to be transformed into a string.

        Returns:
            variable (str): either the original str, a string pulled from a
                one-item list, or the original list.
        """
        if variable is None:
            return 'none'
        elif isinstance(variable, str):
            return variable
        else:
            try:
                return variable[0]
            except TypeError:
                return variable

    """ Public Input/Output Methods """

    def load(self, name = None, file_path = None, folder = None,
             file_name = None, file_format = None):
        """Loads object from file into subclass attribute.

        Args:
            name: name of attribute for the file contents to be stored.
            file_path: a complete file path for the file to be loaded.
            folder: a path to the folder where the file should be loaded from
                (not used if file_path is passed).
            file_name: a string containing the name of the file to be loaded
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                Depot.extensions.
        """
        setattr(self, name, self.depot.load(file_path = file_path,
                                            folder = folder,
                                            file_name = file_name,
                                            file_format = file_format))
        return self

    def save(self, variable = None, file_path = None, folder = None,
             file_name = None, file_format = None):
        """Exports a variable or attribute to disc.

        Args:
            variable: a python object or a string corresponding to a subclass
                attribute which should be saved to disc.
            file_path: a complete file path for the file to be saved.
            folder: a path to the folder where the file should be saved (not
                used if file_path is passed).
            file_name: a string containing the name of the file to be saved
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                Depot.extensions.
        """
        if isinstance(variable, str):
            variable = getattr(self, variable)
        self.depot.save(variable = variable,
                        file_path = file_path,
                        folder = folder,
                        file_name = file_name,
                        file_format = file_format)
        return

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self):
        """Required method that sets default values for a subclass.

        A dict called 'options' should be defined here for subclasses to use
        much of the functionality of SimpleClass.

        Generally, the 'checks' attribute should be set here if the subclass
        wants to make use of related methods.
        """
        pass
        return self

    def edit(self, techniques = None, algorithms = None, options = None):
        """Updates 'options' dictionary with passed arguments.

        Args:
            techniques (str or list): a string name or list of names for keys
                in the 'options' dict.
            algorithms (object or list(object)): siMpLify compatible objects
                which can be integrated in the package framework. If they are
                custom algorithms, they should be subclassed from SimpleStep to
                ensure compatibility.
            options (dict): a dictionary with keys of techniques and values of
                algorithms. This should be passed if the user has already
                combined some or all 'techniques' and 'algorithms' into a dict.
        """
        if not self.exists('options'):
            self.options = {}
        if options:
            self.options.update(options)
        if techniques and algorithms:
            self.options.update(dict(zip(techniques, algorithms)))
        return self

    @abstractmethod
    def publish(self, **kwargs):
        """Required method which creates any objects to be applied to data or
        variables.

        In the case of iterative classes, such as Cookbook, this method should
        construct any plans to be later implemented by the 'read' method.

        Args:
            **kwargs: keyword arguments are not ordinarily included in the
                publish method. But nothing precludes them from being added
                to subclasses.
        """
        pass
        return self