"""
.. module:: base
  :synopsis: contains core classes of siMpLify package.
"""

from abc import ABC, abstractmethod
import csv
from configparser import ConfigParser
from dataclasses import dataclass
from functools import wraps
import glob
from inspect import getfullargspec
from itertools import product
import os
import re
import pickle
import warnings

from more_itertools import unique_everseen
import numpy as np
import pandas as pd
#from tensorflow.test import is_gpu_available

from simplify.core.types import DataTypes, FileTypes


@dataclass
class SimpleClass(ABC):
    """Absract base class for classes in siMpLify package to support a common
    class structure and allow sharing of universal methods.

    To use the class, a subclass must have the following methods:
        draft: a method which sets the default values for the subclass, and
            usually includes the self.options dictionary. If the subclass calls
            super().__post_init__, the 'draft' method is automatically called.
        finalize: a method which, after the user has set all options in the
            preferred manner, constructs the objects which can parse, modify,
            process, analyze, and/or transform data.

    The following methods are not strictly required but should be used if
    the subclass is transforming data or other variable (as opposed to merely
    containing data or variables):
        produce: method which applies the finalized objects to passed data or
            other variables.

    For consistency, methods in subclasses which seek to alter the 'options'
    dict or set parameters should begin with the 'edit_' prefix.

    If the subclass includes boolean attributes of auto_finalize or
    auto_produce, and those attributes are set to True, then the finalize
    and/or produce methods are called when the class is instanced.
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
        # Registers subclass into lists based upon specific subclass needs.
        self._register_subclass()
        # Calls 'finalize' method if it exists and 'auto_finalize' is True.
        if hasattr(self, 'auto_finalize') and self.auto_finalize:
            self.finalize()
            # Calls 'produce' method if it exists and 'auto_produce' is True.
            if hasattr(self, 'auto_produce') and self.auto_produce:
                self.produce()
        return self

    """ Magic Methods """

    def __call__(self, idea, *args, **kwargs):
        """When called as a function, a subclass will return the produce method
        after running __post_init__. Any args and kwargs will be passed to the
        'produce' method.

        Args:
            idea (Idea or str): an instance of Idea or path where an Idea
                configuration file is located must be passed when a subclass is
                called as a function.

        Returns:
            return value of 'produce' method.
        """
        self.idea = idea
        self.auto_finalize = True
        self.auto_produce = False
        self.__post_init__()
        return self.produce(*args, **kwargs)

    def __contains__(self, item):
        """Checks if item is in 'options'.

        Args:
            item (str): item to be searched for in 'options' keys.

        Returns:
            True, if 'item' in 'options' - otherwise False.
        """
        return item in self.options

    def __delitem__(self, item):
        """Deletes item if in 'options' or, if ait is an instance attribute, it
        is assigned a value of None.

        Args:
            item (str): item to be deleted from 'options' or attribute to be
                assigned None.

        Raises:
            KeyError: if item is neither in 'options' or an attribute.
        """
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

        Args:
            attr (str): attribute sought.

        Returns:
            attribute or None, if attribute does not exist.

        Raises:
            AttributeError: if a dunder attribute is sought.
        """
        # Intecepts common dict methods and applies them to 'options' dict.
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
        """Returns item if item is in self.options or is an atttribute.

        Args:
            item (str): item matching dict key or attribute name.

        Returns:
            value for item in 'options', 'item' attribute value, or None if
                neither of those exist.
        """
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
        """Adds item and value to options dictionary.

        Args:
            item (str): 'options' key to be set.
            value (any): corresponding value to be set for 'item' key in
                'options'.
        """
        self.options[item] = value
        return self

    """ Private Methods """

    def _check_depot(self):
        """Adds an Depot instance with default Depot instance as 'depot'
        attribute if one is not passed when subclass is instanced.
        """
        if not hasattr(self, 'depot') or self.depot is None:
            self.depot = Depot(idea = self.idea)
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
        Idea instance. Injects sections of 'idea' to a subclass instance using
        user settings.
        """
        if hasattr(self, 'idea') and isinstance(self.idea, str):
            self.idea = Idea(file_path = self.idea)
        # Adds attributes to class from appropriate sections of the idea.
        sections = ['general']
        if hasattr(self, 'idea_sections') and self.idea_sections:
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
            ingredients: an Ingredients instance, a file path containing a
                DataFrame or Series to add to an Ingredients instance, or
                a folder containing files to be used to compose Ingredients
                DataFrames and/or Series.
        """
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
        """Checks for 'steps' attribute or finds an attribute with the class
        'name' prefix followed by 'steps' if 'steps' does not already exist or
        is None.
        """
        if not hasattr(self, 'steps') or self.steps is None:
            if hasattr(self, self.name + '_steps'):
                if getattr(self, self.name + '_steps') in ['all', 'default']:
                    self.steps = list(self.options.keys())
                else:
                    self.steps = self.listify(getattr(
                        self, self.name + '_steps'))
            else:
                self.steps = []
        else:
            self.steps = self.listify(self.steps)
        if not hasattr(self, 'step') or not self.step:
            self.step = self.steps[0]
        return self

    @classmethod
    def _register_subclass(self):
        """Adds subclass to appropriate list based on attribute in subclass."""
        self._registered_subclasses = {
            'registered_state_subclasses' : 'state_dependent'}
        for subclass_list, attribute in self._registered_subclasses.items():
            if not hasattr(self, subclass_list):
                setattr(self, subclass_list, [])
            if hasattr(self, attribute) and getattr(self, attribute):
                getattr(self, subclass_list).append(self)
        return self

    def _run_checks(self):
        """Checks attributes from 'checks' and runs corresponding methods based
        upon strings stored in 'checks'. Those methods should have the prefix
        '_check_' followed by the string in the attribute 'checks' and have
        no parameters other than 'self'.
        """
        if hasattr(self, 'checks') and self.checks:
            for check in self.checks:
                getattr(self, '_check_' + check)()
        return self

    """ Public Methods """

    def conform(self, step):
        """Sets 'step' attribute to passed 'step' throughout package.

        This method is used to maintain a universal state in the package for
        subclasses that are state dependent. It iterates through any subclasses
        listed in 'registered_state_subclasses' to call their 'conform' methods.

        Args:
            step (str): corresponds to current state in siMpLify package.
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
            iterable (list, DataFrame, or Series, same as passed type): iterable
                with duplicate entries removed.
        """
        if isinstance(iterable, list):
            return list(unique_everseen(iterable))
        elif isinstance(iterable, pd.Series):
            return iterable.drop_duplicates(inplace = True)
        elif isinstance(iterable, pd.DataFrame):
            return iterable.drop_duplicates(inplace = True)

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
            techniques (str or list): a string name or list of names for keys in
            the 'options' dict.
            algorithms (object or list(object)): siMpLify compatible objects
                which can be integrated in the package framework. If they are
                custom algorithms, they should be subclassed from Technique to
                ensure compatibility.
            options (dict): a dictionary with keys of techniques and values of
                algorithms. This should be passed if the user has already
                combined some or all 'techniques' and 'algorithms' into a dict.
        """
        if options:
            self.name_to_type.update(options)
        if techniques and algorithms:
            self.name_to_type.update(dict(zip(techniques, algorithms)))
        return self

    @abstractmethod
    def finalize(self, **kwargs):
        """Required method which creates any objects to be applied to data or
        variables.

        In the case of iterative classes, such as Cookbook, this method should
        construct any plans to be later implemented by the 'produce' method. It
        is roughly equivalent to the scikit-learn fit method.

        Args:
            **kwargs: keyword arguments are not ordinarily included in the
                finalize method. But nothing precludes them from being added
                to subclasses.
        """
        pass
        return self

    @staticmethod
    def listify(variable):
        """Checks to see if the variable is stored in a list. If not, the
        variable is converted to a list or a list of 'none' is created if the
        variable is empty.

        Args:
            variable(str or list): variable to be transformed into a list to
            allow iteration.

        Returns:
            variable(list): either the original list, a string converted to a
                list, or a list containing 'none' as its only item.
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


@dataclass
class Idea(SimpleClass):
    """Loads and/or stores the user's data science idea.

    If configuration settings are imported from a file, Idea creates a nested
    dictionary, converting dictionary values to appropriate datatypes, and
    stores portions of the configuration dictionary as attributes in other
    classes. Idea is based on for python's ConfigParser. It seeks to cure some
    of the most significant shortcomings of the base ConfigParser package:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting items creates verbose code.
        3) It still uses OrderedDict (even though python 3.6+ has automatically
             orders regular dictionaries).

    To use the Idea class, the user can either:
        1) Pass file path and the file will automatically be loaded,
        2) Pass a file name which is located in the current working directory,
            or;
        3) Pass a prebuilt nested dictionary matching the specifications of
        'configuration' for storage in the Idea class.

    Whichever option is chosen, the nested idea dictionary is stored in the
    attribute 'configuration'. Users can store any key/value pairs in a section
    of the 'configuration' dictionary as attributes in a class instance by using
    the 'inject' method.

    If 'infer_types' is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes (str, list, float,
    bool, and int are currently supported)

    For example, if the idea file is as follows:

        [general]
        verbose = True
        seed = 43

        [files]
        source_format = csv
        test_data = True
        test_chunk = 500
        random_test_chunk = True

        [cookbook]
        cookbook_steps = split, reduce, model

    'verbose' and 'file_type' will automatically be added to every siMpLify
    class because they are located in the 'general' section. If a subclass wants
    attributes from the files section, then the following line should appear
    in __post_init__ before calling super().__post_init__:

        self.idea_sections = ['files']

    If the subclass wants the cookbook settings as well, then the code should
    be:
        self.idea_sections = ['files', 'cookbook']

    If that latter code is included, an equivalent to this class will be
    created:

        class FakeClass(object):

            def __init__(self):
                self.verbose = True
                self.seed = 43
                self.source_format = 'csv'
                self.test_data = True
                self.test_chunk = 500
                self.random_test_chunk = True
                self.cookbook_steps = ['split', 'reduce', 'model']
                return self

    Regardless of the idea_sections added, all idea settings can be similarly
    accessed using dict keys. For example:

        self.idea['general']['seed'] # typical dict access technique
                                and
        self.idea['seed'] # if no section or other key is named 'seed'
                            both return 43.


    Because Idea uses ConfigParser, it only allows 2-level dictionaries. The
    desire for accessibility and simplicity dictated this limitation.

    Args:
        configuration(str or dict): either a file path, file name, or two-level
        nested dictionary storing settings. If a file path is provided, A
        nested dict will automatically be created from the file and stored in
            'configuration'. If a file name is provided, Idea will look for it
            in the current working directory and store its contents in
            'configuration'.
        infer_types(bool): variable determines whether values in
            'configuration' are converted to other types (True) or left as
            strings (False).
        auto_finalize(bool): whether to automatically call the 'finalize'
            method when the class is instanced. Unless adding a new source for
            'configuration' settings, this should be set to True.
        auto_produce(bool): whether to automatically call the 'produce' method
            when the class is instanced. Unless adding a new source for
            'configuration' settings, this should be set to True.
    """
    configuration : object = None
    infer_types : bool = True
    auto_finalize : bool = True
    auto_produce : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Magic Methods """

    def __delitem__(self, key):
        """Removes a dictionary section if name matches the name of a section.
        Otherwise, it will remove all entries with name inside the various
        sections of the 'configuration' dictionary.

        Args:
            key(str): the name of the dictionary key or section to be deleted.

        Raises:
            KeyError: if 'key' not in 'configuration'.
        """
        found_value = False
        if key in self.configuration:
            found_value = True
            self.configuration.pop(key)
        else:
            for config_key, config_value in self.configuration.items():
                if key in config_value:
                    found_value = True
                    self.configuration[config_key].pop(key)
        if not found_value:
            error_message = key + ' not found in idea dictionary'
            raise KeyError(error_message)
        return self

    def __getitem__(self, key):
        """Returns a section of the configuration or key within a section.

        Args:
            key(str): the name of the dictionary key for which the value is
                sought.

        Returns:
            dict if 'key' matches a section in 'configuration'. If 'key' matches
                a key within a section, the value, which can be any of the
                supported datatypes is returned. If no match is found an empty
                dict is returned.
        """
        found_value = False
        if key in self.configuration:
            found_value = True
            return self.configuration[key]
        else:
            for config_key, config_value in self.configuration.items():
                if key in config_value:
                    found_value = True
                    return self.configuration[config_key]
        if not found_value:
            return {}

    def __iter__(self):
        """Returns iterable configuration dict items()."""
        return self.items()

    def __setitem__(self, section, nested_dict):
        """Creates a new subsection in a specified section of the configuration
        nested dictionary.

        Args:
            section(str): a string naming the section of the configuration
                dictionary.
            nested_dict(dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'section' isn't a str or 'nested_dict' isn't a dict.
        """
        if isinstance(section, str):
            if isinstance(nested_dict, dict):
                self.configuration.update({section, nested_dict})
            else:
                error_message = 'nested_dict must be dict type'
                raise TypeError(error_message)
        else:
            error_message = 'section must be str type'
            raise TypeError(error_message)
        return self

    """ Private Methods """

    def _check_configuration(self):
        """Checks the datatype of 'configuration' and sets 'technique' to
        properly finalize 'configuration'.

        Raises:
            AttributeError if 'configuration' attribute is neither a file path
                dict, nor Idea instance.
        """
        if self.configuration:
            if isinstance(self.configuration, str):
                if '.ini' in self.configuration:
                    self.technique = 'ini_file'
                elif '.py' in self.configuration:
                    self.technique = 'py_file'
                else:
                    error = 'configuration file must be .py or .ini file'
                    raise FileNotFoundError(error)
                if not os.path.isfile(os.path.abspath(self.configuration)):
                    self.configuration = os.path.join(os.getcwd(),
                                                      self.configuration)
            elif not isinstance(self.configuration, dict):
                error = 'configuration must be dict or file path'
                raise TypeError(error)
        else:
            error = 'configuration dict or path needed to instance Idea'
            raise AttributeError(error)
        return self

    def _create_from_ini(self):
        """Creates a configuration dictionary from an .ini file."""
        if os.path.isfile(self.configuration):
            configuration = ConfigParser(dict_type = dict)
            configuration.optionxform = lambda option : option
            configuration.read(self.configuration)
            self.configuration = dict(configuration._sections)
        else:
            error = 'configuration file ' + self.configuration + ' not found'
            raise FileNotFoundError(error)
        return self

    def _create_from_py(self):
        """Creates a configuration dictionary from an .py file.

        Todo:
            Add .py file implementation.
        """
        pass
        return self

    def _infer_types(self):
        """If 'infer_types' is True, all dictionary values in 'configuration'
        are converted to the appropriate datatype.
        """
        if self.infer_types:
            for section, nested_dict in self.configuration.items():
                for key, value in nested_dict.items():
                    self.configuration[section][key] = self._typify(value)
        return self

    def _inject_base(self):
        """Injects parent class, SimpleClass with this Idea instance so that
        the instance is available to other files in the siMpLify package.
        """
        setattr(SimpleClass, 'idea', self)
        return self

    def _typify(self, variable):
        """Converts str to appropriate, supported datatype.

        The method converts strings to list (if ', ' is present), int, float, or
        bool datatypes based upon the content of the string. If no alternative
        datatype is found, the variable is returned in its original form.

        Args:
            variable(str): string to be converted to appropriate datatype.

        Returns:
            variable(str, list, int, float, or bool): converted variable.
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

    """ Public Methods """

    def draft(self):
        """Sets options to create 'configuration' dict'."""
        # Sets options for creating 'configuration'.
        self.options = {'py_file' : self._create_from_py,
                        'ini_file' : self._create_from_ini,
                        'dict' : None}
        return self

    def finalize(self):
        """Prepares instance of Idea by checking passed configuration parameter.
        """
        self._check_configuration()
        return self

    def inject(self, instance, sections, override = False):
        """Stores the section or sections of the 'configuration' dictionary in
        the passed class instance as attributes to that class instance.

        Args:
            instance(object): either a class instance or class to which
                attributes should be added.
            sections(str or list): the sections of the configuration dictionary
            which should have key, value pairs added as attributes to
                instance.
            override (bool): if True, even existing attributes in instance will
                be replaced by configuration dictionary items. If False, current
                values in those similarly-named attributes will be maintained.

        Returns:
            instance with attributes added.
        """
        for section in self.listify(sections):
            for key, value in self.configuration[section].items():
                if not hasattr(instance, key) or override:
                    setattr(instance, key, value)
        return instance

    def items(self):
        """Returns items of 'configuration' dict to mirror dict functionality.

        This method is also accessed if the user attempts to iterate the class.
        """
        return self.configuration.items()

    def keys(self):
        """Returns keys (section names) of 'configuration' dict to mirror dict
        functionality.
        """
        return self.configuration.keys()

    def produce(self):
        """Creates configuration setttings and injects Idea into SimpleClass.
        """
        if self.options[self.technique]:
            self.options[self.technique]()
        self._infer_types()
        self._inject_base()
        return self

    def update(self, new_settings):
        """Adds new settings to the configuration dictionary.

        Args:
           new_settings(dict, str, or Idea): can either be a dictionary or Idea
           object containing new attribute, value pairs or a string
               containing a file path from which new configuration options can
               be found.

        Raises:
            TypeError: if 'new_settings' is neither a dict, str, or Idea
                instance.
        """
        if isinstance(new_settings, dict):
            self.configuration.update(new_settings)
        elif isinstance(new_settings, str):
            self.configuration.update(
                    self._create_configuration(file_path = new_settings))
        elif (hasattr(new_settings, 'configuration')
                and isinstance(new_settings.configuration, dict)):
            self.configuration.update(new_settings.configuration)
        else:
            error_message = 'new_options must be dict, Idea instance, or path'
            raise TypeError(error_message)
        return self

    def values(self):
        """Returns values (sections) of 'configuration' dict to mirror dict
        functionality.
        """
        return self.configuration.values()

@dataclass
class Depot(SimpleClass):
    """Manages files and folders for the Creates and stores dynamic and static
    file paths, properly formats
    files for import and export, and allows loading and saving of siMpLify,
    pandas, and numpy objects in set folders.

    Args:
        root_folder(str): the complete path from which the other paths and
        folders     used by Depot should be created.
        data_folder(str): the data subfolder name or a complete path if the
        'data_folder' is not off of 'root_folder'.
        results_folder(str): the results subfolder name or a complete path if
            the 'results_folder' is not off of 'root_folder'.
        datetime_naming(bool): whether the date and time should be used to
            create experiment subfolders (so that prior results are not
            overwritten).
        auto_finalize(bool): whether to automatically call the 'finalize' method
            when the class is instanced. Unless making major changes to the
            file structure (beyond the 'root_folder', 'data_folder', and
            'results_folder' parameters), this should be set to True.
        auto_produce(bool): whether to automatically call the 'produce' method
            when the class is instanced.
        state_dependent(bool): whether this class is depending upon the current
            state in the siMpLify package. Unless the user is radically changing
            the way siMpLify works, this should be set to True.
    """
    root_folder : str = ''
    data_folder : str = 'data'
    results_folder : str = 'results'
    datetime_naming : bool = True
    auto_finalize : bool = True
    auto_produce : bool = True
    state_dependent : bool = True

    def __post_init__(self):
        # Adds additional section of idea to be injected as local attributes.
        self.idea_sections = ['files']
        super().__post_init__()
        return self

    """ Private Methods """

    def _add_branch(self, root_folder, subfolders):
        """Creates a branch of a folder tree and stores each folder name as
        a local variable containing the path to that folder.

        Args:
            root_folder(str): the folder from which the tree branch should be
                created.
            subfolders(str or list): subfolder names to form the tree branch.
        """
        for subfolder in self.listify(subfolders):
            temp_folder = self.create_folder(folder = root_folder,
                                             subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
            root_folder = temp_folder
        return self

    def _check_boolean_out(self, variable):
        """Either leaves boolean values as True/False or changes values to 1/0
        based on user settings.

        Args:
            variable(DataFrame or Series): pandas DataFrame or Series with some
            boolean values.

        Returns:
            variable(DataFrame or Series): either the original pandas data or
                the dataset with True/False converted to 1/0.
        """
        # Checks whether True/False should be exported in data files. If
        # 'boolean_out' is set to False, 1/0 are used instead.
        if hasattr(self, 'boolean_out') and self.boolean_out == False:
            variable.replace({True : 1, False : 0}, inplace = True)
        return variable

    def _check_file_name(self, file_name, io_status = None):
        """Checks passed file_name to see if it exists. If not, depending
        upon the io_status, a default file_name is returned.

        Args:
            file_name(str): file name (without extension).
            io_status(str): either 'import' or 'export' based upon whether the
                user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file name.
        """
        if file_name:
            return file_name
        elif io_status == 'import':
            return self.file_in
        elif io_status == 'export':
            return self.file_out

    def _check_file_format(self, file_format = None, io_status = None):
        """Checks value of local file_format variable. If not supplied, the
        default from the Idea instance is used based upon whether import or
        export methods are being used. If the Idea options don't exist,
        '.csv' is returned.

        Args:
            file_format(str): one of the supported file types in 'extensions'.
            io_status(str): either 'import' or 'export' based upon whether the
            user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file format.
        """
        if file_format:
            return file_format
        elif io_status == 'import':
            return self.format_in
        elif io_status == 'export':
            return self.format_out
        else:
            return 'csv'

    def _check_folder(self, folder, io_status = None):
        """Checks if folder is a full path or string matching an attribute.
        If no folder name is provided, a default value is used.

        Args:
            folder: a string either containing a folder path or the name of an
                attribute containing a folder path.
            io_status(str): either 'import' or 'export' based upon whether the
            user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file folder path.
        """
        if folder and os.path.isdir(folder):
            return folder
        elif folder and isinstance(folder, str):
            return getattr(self, folder)
        elif io_status == 'import':
            return self.folder_in
        elif io_status == 'export':
            return self.folder_out

    def _check_kwargs(self, variables_to_check, passed_kwargs):
        """Checks kwargs to see which ones are required for the particular
        method and/or substitutes default values if needed.

        Args:
            variables_to_check(list): variables to check for values.
            passed_kwargs(dict): kwargs passed to method.

        Returns:
            new_kwargs(dict): kwargs with only relevant parameters.
        """
        new_kwargs = passed_kwargs
        for variable in variables_to_check:
            if not variable in passed_kwargs:
                if variable in self.default_kwargs:
                    new_kwargs.update(
                            {variable : self.default_kwargs[variable]})
                elif hasattr(self, variable):
                    new_kwargs.update({variable : getattr(self, variable)})
        return new_kwargs

    def _check_root_folder(self):
        """Checks if 'root_folder' exists on disc. If not, it is created."""
        if self.root_folder:
            if os.path.isdir(self.root_folder):
                self.root = self.root_folder
            else:
                self.root = os.path.abspath(self.root_folder)
        else:
            self.root = os.path.join('..', '..')
        return self

    def _get_file_format(self, io_status):
        """Returns appropriate file format based on 'step' and 'io_status'.

        Args:
            io_status(str): either 'import' or 'export' based upon whether the
            user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file format.
        """
        if (self.options[self.step][self.options_index[io_status]]
                in ['raw']):
            return self.source_format
        elif (self.options[self.step][self.options_index[io_status]]
                in ['interim']):
            return self.interim_format
        elif (self.options[self.step][self.options_index[io_status]]
                in ['processed']):
            return self.final_format

    def _inject_base(self):
        """Injects parent class with this Depot instance so that the instance is
        available to other files in the siMpLify package.
        """
        SimpleClass.depot = self
        return self

    def _load_csv(self, file_path, **kwargs):
        """Loads csv file into a pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        additional_kwargs = ['encoding', 'index_col', 'header', 'usecols',
                             'low_memory']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows' : self.test_chunk})
        variable = pd.read_csv(file_path, **kwargs)
        return variable

    def _load_excel(self, file_path, **kwargs):
        """Loads Excel file into a pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        additional_kwargs = ['index_col', 'header', 'usecols']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows' : self.test_chunk})
        variable = pd.read_excel(file_path, **kwargs)
        return variable

    def _load_feather(self, file_path, **kwargs):
        """Loads feather file into pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        return pd.read_feather(file_path, nthreads = -1, **kwargs)

    def _load_h5(self, file_path, **kwargs):
        """Loads hdf5 with '.h5' extension into pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        return self._load_hdf(file_path, **kwargs)

    def _load_hdf(self, file_path, **kwargs):
        """Loads hdf5 file into pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        additional_kwargs = ['columns']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize' : self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns' : kwargs['usecols']})
            kwargs.pop('usecols')
        return pd.read_hdf(file_path, **kwargs)

    def _load_json(self, file_path, **kwargs):
        """Loads json file into pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        additional_kwargs = ['encoding', 'columns']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize' : self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns' : kwargs['usecols']})
            kwargs.pop('usecols')
        return pd.read_json(file_path = file_path, **kwargs)

    def _load_pickle(self, file_path, **kwargs):
        """Returns an unpickled python object.

        Args:
            file_path: complete file path of file.

        Returns:
            variable(object): pickled object loaded from disc.
        """
        return pickle.load(open(file_path, 'rb'))

    def _load_png(self, file_path, **kwargs):
        """Although png files are saved by siMpLify, they cannot be loaded.

        Raises:
            NotImplementedError: if called.
        """
        error = 'loading .png files is not supported'
        raise NotImplementedError(error)

    def _load_text(self, file_path, **kwargs):
        """Loads text file with python reader.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(str): string loaded from disc.
        """
        return self._load_txt(file_path = file_path, **kwargs)

    def _load_txt(self, file_path, **kwargs):
        """Loads text file with python reader.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(str): string loaded from disc.
        """
        with open(file_path, mode = 'r', errors = 'ignore',
                  encoding = self.file_encoding) as a_file:
            return a_file.read()

    def _make_folder(self, folder):
        """Creates folder if it doesn't already exist.

        Args:
            folder(str): the path of the folder.
        """
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _save_csv(self, variable, file_path, **kwargs):
        """Saves csv file to disc.

        Args:
            variable(Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.to_csv(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_excel(self, variable, file_path, **kwargs):
        """Saves Excel file to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.excel(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_feather(self, variable, file_path, **kwargs):
        """Saves feather file to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.reset_index(inplace = True)
        variable.to_feather(file_path, **kwargs)
        return

    def _save_h5(self, variable, file_path, **kwargs):
        """Saves hdf file with .h5 extension to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.to_hdf(file_path, **kwargs)
        return

    def _save_hdf(self, variable, file_path, **kwargs):
        """Saves hdf file to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.to_hdf(file_path, **kwargs)
        return

    def _save_json(self, variable, file_path, **kwargs):
        """Saves json file to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.to_json(file_path, **kwargs)
        return

    def _save_pickle(self, variable, file_path, **kwargs):
        """Pickles file and saves it to disc.

        Args:
            variable(object): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        pickle.dump(variable, open(file_path, 'wb'))
        return

    def _save_png(self, variable, file_path, **kwargs):
        """Saves png file to disc.

        Args:
            variable(matplotlib object): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.savefig(file_path, bbox_inches = 'tight')
        variable.close()
        return

    def add_tree(self, folder_tree):
        """Adds a folder tree to disc with corresponding attributes to the
        Depot instance.

        Args:
            folder_tree(dict): a folder tree to be created with corresponding
            attributes to the Depot instance.
        """
        for folder, subfolders in folder_tree.items():
            self._add_branch(root_folder = folder, subfolders = subfolders)
        return self

    def create_batch(self, folder = None, file_format = None,
                    include_subfolders = True):
        """Creates a list of paths in 'folder_in' based upon 'file_format'.

        If 'include_subfolders' is True, subfolders are searched as well for
        matching 'file_format' files.

        Args:
            folder(str): path of folder or string corresponding to class
                attribute with path.
            file_format(str): file format name.
            include_subfolders(bool):  whether to include files in subfolders
                when creating a batch.
        """
        folder = self._check_folder(folder = folder)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'import')
        extension = self.extensions[file_format]
        return glob.glob(os.path.join(folder, '**', '*' + extension),
                         recursive = include_subfolders)

    def create_folder(self, folder, subfolder = None):
        """Creates folder path from component parts.

        Args:
            folder(str): path of folder or string corresponding to class
                attribute containing folder path.
            subfolder(str): subfolder name to be created off of folder.
        """
        if subfolder:
            if folder and os.path.isdir(folder):
                folder = os.path.join(folder, subfolder)
            else:
                folder = os.path.join(getattr(self, folder), subfolder)
        self._make_folder(folder = folder)
        return folder

    def create_path(self, folder = None, file_name = None, file_format = None,
                    io_status = None):
        """Creates file path from component parts.

        Args:
            folder(str): path of folder or string corresponding to class
                attribute containing folder path.
            file_name(str): file name without extension.
            file_format(str): file format name from 'extensions' dict.
            io_status: 'import' or 'export' indicating which direction the path
                is used for storing files (only needed to use defaults when
                other parameters are not provided).
            """
        folder = self._check_folder(folder = folder,
                                    io_status = io_status)
        file_name = self._check_file_name(file_name = file_name,
                                          io_status = io_status)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = io_status)
        extension = self.extensions[file_format]
        if file_name == 'glob':
            file_path = self.create_batch(folder = folder,
                                          file_format = file_format)
        else:
            file_path = os.path.join(folder, file_name) + extension
        return file_path

    def draft(self):
        """Creates default folder and file dicts."""
        # Initializes dict with file format names and corresponding file
        # extensions.
        self.extensions = FileTypes()
        # Creates list of default subfolders from 'data_folder' to create.
        self.data_subfolders = ['raw', 'interim', 'processed', 'external']
        # Creates default parameters when they are not passed as kwargs to
        # methods in the class.
        self.default_kwargs = {'index' : False,
                               'header' : None,
                               'low_memory' : False,
                               'dialect' : 'excel',
                               'usecols' : None,
                               'columns' : None,
                               'nrows' : None,
                               'index_col' : False}
        # Creates options dict with keys as names of stages in siMpLify and the
        # values as 2-item lists with the first item being the default import
        # folder and the second being the default export folder.
        self.options = {'sow' : ['raw', 'raw'],
                        'reap' : ['raw', 'interim'],
                        'clean' : ['interim', 'interim'],
                        'bale' : ['interim', 'interim'],
                        'deliver' : ['interim', 'processed'],
                        'cook' : ['processed', 'processed'],
                        'critic' : ['processed', 'processed']}
        # Sets index for import and export folders in 'options' dict.
        self.options_index = {'import' : 0,
                              'export' : 1}
        # Sets default folders to use for imports and exports that are not
        # data containers. The keys are based upon 'name' attributes of classes.
        self.class_options = {'ingredients' : 'default',
                              'datatypes' : 'recipe',
                              'cookbook' : 'experiment',
                              'recipe' : 'recipe',
                              'almanac' : 'harvesters',
                              'harvest' : 'harvesters',
                              'review' : 'experiment',
                              'visualizer' : 'recipe',
                              'evaluator' : 'recipe',
                              'summarizer' : 'recipe',
                              'comparer' : 'experiment'}
        return self

    def edit_file_formats(self, file_format, extension, load_method,
                          save_method):
        """Adds or replaces a file extension option.

        Args:
            file_format(str): string name of the file_format.
            extension(str): file extension (without period) to be used.
            load_method(method): a method to be used when loading files of the
            passed file_format.
            save_method(method): a method to be used when saving files of the
            passed file_format.
        """
        self.extensions.update({file_format : extension})
        if isinstance(load_method, str):
            setattr(self, '_load_' + file_format, '_load_' + load_method)
        else:
            setattr(self, '_load_' + file_format, load_method)
        if isinstance(save_method, str):
            setattr(self, '_save_' + file_format, '_save_' + save_method)
        else:
            setattr(self, '_save_' + file_format, save_method)
        return self

    def edit_folders(self, root_folder, subfolders):
        """Adds a list of subfolders to an existing root_folder.

        Args:
            root_folder(str): path of folder where subfolders should be created.
            subfolders(str or list): subfolder names to be created.
        """
        for subfolder in self.listify(subfolders):
            temp_folder = self.create_folder(folder = root_folder,
                                             subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
        return self

    def finalize(self):
        """Creates data and results folders as well as other default subfolders.
        """
        self._check_root_folder()
        self.edit_folders(root_folder = self.root,
                         subfolders = [self.data_folder, self.results_folder])
        self.edit_folders(root_folder = self.data,
                         subfolders = self.data_subfolders)
        return self

    def initialize_writer(self, file_path, columns, encoding = None,
                          dialect = 'excel'):
        """Initializes writer object for line-by-line exporting to a .csv file.

        Args:
            file_path(str): a complete path to the file being written to.
            columns(list): column names to be added to the first row of the
                file as column headers.
            encoding(str): a python encoding type.
            dialect(str): the specific type of csv file created.
        """
        if not columns:
            error = 'initialize_writer requires columns as a list type'
            raise TypeError(error)
        with open(file_path, mode = 'w', newline = '',
                  encoding = self.file_encoding) as self.output_series:
            self.writer = csv.writer(self.output_series, dialect = dialect)
            self.writer.writerow(columns)
        return self

    def inject(self, instance, sections, override = True):
        """Stores the default paths in the passed instance.

        Args:
            instance(object): either a class instance or class to which
                attributes should be added.
            sections(list): attributes to be added to passed class. Data import
                and export paths are automatically added.
            override(bool): if True, existing attributes in instance will be
                replaced by items from this class.

        Returns:
            No value is returned, but passed instance is now injected with
            selected attributes.
        """
        instance.data_in = self.path_in
        instance.data_out = self.path_out
        for section in self.listify(sections):
            if hasattr(self, section + '_in') and override:
                setattr(instance, section + '_in',
                        getattr(self, section + '_in'))
                setattr(instance, section + '_out',
                        getattr(self, section + '_out'))
            elif override:
                setattr(instance, section, getattr(self, section))
        return

    def iterate(self, plans, ingredients = None, return_ingredients = True):
        """Iterates through a list of files contained in self.batch and
        applies the plans created by a Planner method (or subclass).

        Args:
            plans(list): list of plan types (Recipe, Harvest, etc.)
            ingredients(Ingredients): an instance of Ingredients or subclass.
            return_ingredients(bool): whether ingredients should be returned by
            this method.

        Returns:
            If 'return_ingredients' is True: an in instance of Ingredients.
            If 'return_ingredients' is False, no value is returned.
        """
        if ingredients:
            for file_path in self.batch:
                ingredients.source = self.load(file_path = file_path)
                for plan in plans:
                    ingredients = plan.produce(ingredients = ingredients)
            if return_ingredients:
                return ingredients
            else:
                return self
        else:
            for file_path in self.batch:
                for plan in plans:
                    plan.produce()
            return self

    def load(self, file_path = None, folder = None, file_name = None,
             file_format = None, **kwargs):
        """Imports file by calling appropriate method based on file_format. If
        the various arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            file_path(str): a complete file path for the file to be loaded.
            folder(str): a path to the folder from where file is located
                (not used if file_path is passed).
            file_name(str): contains the name of the file to be loaded
                without the file extension (not used if file_path is passed).
            file_format(str): a string matching one the file formats in
                'extensions'.
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        Returns:
            Depending upon method used for appropriate file format, a new
                variable of a supported type is returned.

        Raises:
            TypeError: if file_path is not a string (likely a glob list)
        """
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'import')
        if not file_path:
            file_path = self.create_path(folder = folder,
                                         file_name = file_name,
                                         file_format = file_format,
                                         io_status = 'import')
        if isinstance(file_path, str):
            return getattr(self, '_load_' + file_format)(file_path = file_path,
                                                         **kwargs)
        elif isinstance(file_path, list):
            error = 'file_path is a glob list - use iterate instead'
            raise TypeError(error)
        else:
            return None

    def produce(self):
        """Injects Depot instance into base SimpleClass."""
        self._inject_base()
        return self

    def save(self, variable, file_path = None, folder = None, file_name = None,
             file_format = None, **kwargs):
        """Exports file by calling appropriate method based on file_format. If
        the various arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            variable(any): the variable being exported.
            file_path(str): a complete file path for the file to be saved.
            folder(Str): path to the folder where the file should be saved (not
                used if file_path is passed).
            file_name(str): a string containing the name of the file to be saved
                without the file extension (not used if file_path is passed).
            file_format(str): a string matching one the file formats in
                'extensions'.
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.
        """
        # Changes boolean values to 1/0 if self.boolean_out = False
        variable = self._check_boolean_out(variable = variable)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'export')
        if not file_path:
            file_path = self.create_path(folder = folder,
                                         file_name = file_name,
                                         file_format = file_format,
                                         io_status = 'export')
        getattr(self, '_save_' + file_format)(variable, file_path, **kwargs)
        return

    """ Properties """

    @property
    def file_in(self):
        """Returns the data input file name, or folder containing data in
        multiple files.
        """
        if self.options[self.step][0] in ['raw']:
            return self.folder_in
        else:
            return list(self.options.keys())[list(
                    self.options.keys()).index(self.step) - 1] + 'ed_data'

    @property
    def file_out(self):
        """Returns the data output file name, or folder containing where
        multiple files should be saved.
        """
        if self.options[self.step][1] in ['raw']:
            return self.folder_out
        else:
            return self.step + 'ed_data'

    @property
    def folder_in(self):
        """Returns folder where the data input file is located."""
        return getattr(self, self.options[self.step])[0]

    @property
    def folder_out(self):
        """Returns folder where the data output file is located."""
        return getattr(self, self.options[self.step])[1]

    @property
    def format_in(self):
        """Returns file format of input data file."""
        return self._get_file_format(io_status = 'import')

    @property
    def format_out(self):
        """Returns file format of output data file."""
        return self._get_file_format(io_status = 'export')

    @property
    def path_in(self):
        """Returns full file path of input data file."""
        return self.create_path(io_status = 'import')

    @property
    def path_out(self):
        """Returns full file path of output data file."""
        return self.create_path(io_status = 'export')


@dataclass
class Ingredients(SimpleClass):
    """Stores pandas DataFrames and Series with related information about those
    data containers and additional methods to apply to them.

    Ingredients uses pandas DataFrames or Series for all data storage, but it
    utilizes faster numpy methods where possible to increase performance.

    DataFrames and Series stored in ingredients can be imported and exported
    using the 'load' and 'save' methods in a class instance.

    Ingredients adds easy-to-use methods for common feature engineering
    techniques. In addition, any user function can be applied to a DataFrame
    or Series contained in Ingredients by using the 'apply' method (mirroring
    the functionality of the pandas method).

    Args:
        df(DataFrame, Series, or str): either a pandas data object or a string
            containing the complete path where a supported file type with data
            is located. This argument should be passed if the user has a
            pre-existing dataset and is not creating a new dataset with Farmer.
        default_df(str): the current default DataFrame or Series attribute name
        that will be used when a specific DataFrame is not passed to a
            method within the class. The default value is initially set to
            'df'. The decorator check_df will look to the default_df to pick
            the appropriate DataFrame in situations where no DataFrame is passed
            to a method.
        x, y, x_train, y_train, x_test, y_test, x_val, y_val(DataFrames,
            Series, or file paths): These need not be passed when the class is
            instanced. They are merely listed for users who already have divided
            datasets and still wish to use the siMpLify package.
        datatypes(dict): contains column names as keys and datatypes for values
            for columns in a DataFrames or Series. Ingredients assumes that all
            data containers within the instance are related and share a pool of
            column names and types.
        prefixes(dict): contains column prefixes as keys and datatypes for
            values for columns in a DataFrames or Series. Ingredients assumes
            that all data containers within the instance are related and share a
            pool of column names and types.
        auto_finalize(bool): whether 'finalize' method should be called when the
        class is instanced. This should generally be set to True.
        state_dependent(bool): whether this class is depending upon the current
            state in the siMpLify package. Unless the user is radically changing
            the way siMpLify works, this should be set to True.
    """

    df : object = None
    default_df : str = 'df'
    x : object = None
    y : object = None
    x_train : object = None
    y_train : object = None
    x_test : object = None
    y_test : object = None
    x_val : object = None
    y_val : object = None
    datatypes : object = None
    prefixes : object = None
    auto_finalize : bool = True
    state_dependent : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Magic Methods """

    def __getattr__(self, attr):
        """Returns values from column datatypes, column datatype dictionary,
        and section prefixes dictionary.

        Args:
            attr(str): attribute sought.

        Returns:
            If 'attr' is the name of a DataFrame or Series beginning with 'x' or
                'y'), a DataFrame or Series is returned based upon the current
                mapping in the 'options' dict.
            If 'attr' is the name of a datatype, a list of columns that have
                that datatype is returned. In addition 'numerics' combines float
                and int columns into one list.
            If 'attr' is related to one of the Cookbook methods ('scalers',
                'encoders', 'mixers'), a list of columns stored in that
                attribute name or default for that method type.

        Raises:
            AttributeError: if user attempts to access dunder method.
        """
        if attr in ['x', 'y', 'x_train', 'y_train', 'x_test', 'y_test']:
            return self.__dict__[self.options[attr]]
        elif attr in ['booleans', 'floats', 'integers', 'strings',
                      'categoricals', 'lists', 'datetimes', 'timedeltas']:
            return self._get_columns_by_type(attr[:-1])
        elif attr in ['numerics']:
            return (self._get_columns_by_type('float')
                    + self._get_columns_by_type('integer'))
        elif (attr in ['scalers', 'encoders', 'mixers']
              and attr not in self.__dict__):
            return getattr(self, '_get_default_' + attr)()
        elif attr in self.__dict__:
            return self.__dict__[attr]
        elif attr.startswith('__') and attr.endswith('__'):
            error = 'Access to magic methods not permitted through __getattr__'
            raise AttributeError(error)
        else:
            error = attr + ' not found in ' + self.__class__.__name__
            raise AttributeError(error)

    def __setattr__(self, attr, value):
        """Sets values in column datatypes, column datatype dictionary, and
        section prefixes dictionary.

        Args:
            attr(str): string of attribute name to be set.
            value(any): value of the set attribute.
        """
        if attr in ['booleans', 'floats', 'integers', 'strings',
                    'categoricals', 'lists', 'datetimes', 'timedeltas']:
            self.__dict__['datatypes'].update(
                    dict.fromkeys(self.listify(
                            self._all_datatypes[attr[:-1]]), value))
            return self
        else:
            self.__dict__[attr] = value
            return self

    """ Decorators """

    def check_df(method):
        """Decorator which automatically uses the default DataFrame if one
        is not passed to the decorated method.

        Args:
            method(method): wrapped method.

        Returns:
            df(DataFrame or Series): if the passed 'df' parameter was None,
                the attribute named by 'default_df' will be passed. Otherwise,
                df will be passed to the wrapped method intact.
        """
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(method)
            unpassed_args = argspec.args[len(args):]
            if 'df' in argspec.args and 'df' in unpassed_args:
                kwargs.update({'df' : getattr(self, self.default_df)})
            return method(self, *args, **kwargs)
        return wrapper

    def column_list(method):
        """Decorator which creates a complete column list from kwargs passed
        to wrapped method.

        Args:
            method(method): wrapped method.

        Returns:
            new_kwargs(dict): 'columns' parameter has items from 'columns',
                'prefixes', and 'mask' parameters combined into a single list
                of column names using the 'create_column_list' method.
        """
        # kwargs names to use to create finalized 'columns' argument
        arguments_to_check = ['columns', 'prefixes', 'mask']
        new_kwargs = {}
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(method)
            unpassed_args = argspec.args[len(args):]
            if ('columns' in unpassed_args
                    and 'prefixes' in unpassed_args
                    and 'mask' in unpassed_args):
                columns = list(self.datatypes.keys())
            else:
                for argument in arguments_to_check:
                    if argument in kwargs:
                        new_kwargs[argument] = kwargs[argument]
                    else:
                        new_kwargs[argument] = None
                    if argument in ['prefixes', 'mask'] and argument in kwargs:
                        del kwargs[argument]
                columns = self.create_column_list(**new_kwargs)
                kwargs.update({'columns' : columns})
            return method(self, **kwargs)
        return wrapper

    """ Private Methods """

    def _check_columns(self, columns = None):
        """Returns self.datatypes if columns doesn't exist.

        Args:
            columns(list): column names.

        Returns:
            if columns is not None, returns columns, otherwise, the keys of
                the 'datatypes' attribute is returned.
        """
        return columns or list(self.datatypes.keys())

    @check_df
    def _crosscheck_columns(self, df = None):
        """Removes any columns in datatypes dictionary, but not in df.

        Args:
            df(DataFrame or Series): pandas object with column names to
                crosscheck.
        """
        for column in self.datatypes.keys():
            if column not in df.columns:
                del self.datatypes[column]
        return self

    def _get_columns_by_type(self, datatype):
        """Returns list of columns of the specified datatype.

        Args:
            datatype(str): string matching datatype in 'all_datatypes'.

        Returns:
            list of columns matching the passed 'datatype'.
        """
        return [k for k, v in self.datatypes.items() if v == datatype]

    def _get_default_encoders(self):
        """Returns list of categorical columns."""
        return self.categoricals

    def _get_default_mixers(self):
        """Returns an empty list of mixers."""
        return []

    def _get_default_scalers(self):
        """Returns all numeric columns."""
        return self.integers + self.floats

    def _initialize_datatypes(self, df = None):
        """Initializes datatypes for columns of pandas DataFrame or Series if
        not already provided.

        Args:
            df(DataFrame): for datatypes to be determined.
        """
        check_order = [df, getattr(self, self.default_df), self.x,
                       self.x_train]
        for _data in check_order:
            if isinstance(_data, pd.DataFrame) or isinstance(_data, pd.Series):
                if not self.datatypes:
                    self.infer_datatypes(df = _data)
                else:
                    self._crosscheck_columns(df = _data)
                break
        return self

    def _remap_dataframes(self, data_to_use = None):
        """Remaps DataFrames returned by various properties of Ingredients so
        that methods and classes of siMpLify can use the same labels for
        analyzing the Ingredients DataFrames.

        Args:
            data_to_use(str): corresponding to a class property indicating
                which set of data is to be returned when the corresponding
                property is called.
        """
        # Creates flag to track if iterable class is using the validation set.
        self.using_validation_set = False
        # Sets data_to_use to default 'train_test' if no argument passed.
        if not data_to_use:
            data_to_use = 'train_test'
        # Sets values in self.options which contains the mapping for class
        # attributes and DataFrames as determined in __getattr__.
        if data_to_use == 'train_test':
            self.options = self.default_options.copy()
        elif data_to_use == 'train_val':
            self.options['x_test'] = 'x_val'
            self.options['y_test'] = 'y_val'
            self.using_validation_set = True
        elif data_to_use == 'full':
            self.options['x_train'] = 'x'
            self.options['y_train'] = 'y'
            self.options['x_test'] = 'x'
            self.options['y_test'] = 'y'
        elif data_to_use == 'train':
            self.options['x'] = 'x_train'
            self.options['y'] = 'y_train'
        elif data_to_use == 'test':
            self.options['x'] = 'x_test'
            self.options['y'] = 'y_test'
        elif data_to_use == 'val':
            self.options['x'] = 'x_val'
            self.options['y'] = 'y_val'
            self.using_validation_set = True
        return self

    """ Public Methods """

    def draft(self):
        """Sets defaults for Ingredients when class is instanced."""
        # Declares dictionary of DataFrames contained in Ingredients to allow
        # temporary remapping of attributes in __getattr__. __setattr does
        # not use this mapping.
        self.options = {'x' : 'x',
                        'y' : 'y',
                        'x_train' : 'x_train',
                        'y_train' : 'y_train',
                        'x_test' : 'x_test',
                        'y_test' : 'y_test',
                        'x_val' : 'x_val',
                        'y_val' : 'y_val'}
        # Copies 'options' so that original mapping is preserved.
        self.default_options = self.options.copy()
        # Creates object for all available datatypes.
        self.all_datatypes = DataTypes()
        # Creates 'datatypes' and 'prefixes' dicts if they don't exist.
        if not self.datatypes:
            self.datatypes = {}
        if not self.prefixes:
            self.prefixes = {}
        # Maps class properties to appropriate DataFrames using the default
        # train_test setting.
        self._remap_dataframes(data_to_use = 'train_test')
        # Initializes a list of dropped column names so that users can track
        # which features are omitted from analysis.
        self.dropped_columns = []
        return self

    @check_df
    def add_unique_index(self, df = None, column = 'index_universal',
                         make_index = False):
        """Creates a unique integer index for each row.

        Args:
            df(DataFrame): pandas object for index column to be added.
            column(str): contains the column name for the index.
            make_index(bool): boolean value indicating whether the index column
                should be made the actual index of the DataFrame.

        Raises:
            TypeError: if 'df' is not a DataFrame (usually because a Series is
                passed).
        """
        if isinstance(df, pd.DataFrame):
            df[column] = range(1, len(df.index) + 1)
            self.datatypes.update({column, int})
            if make_index:
                df.set_index(column, inplace = True)
        else:
            error = 'To add an index, df must be a pandas DataFrame.'
            TypeError(error)
        return self

    @check_df
    def apply(self, df = None, func = None, **kwargs):
        """Allows users to pass a function to Ingredients instance which will
        be applied to the passed DataFrame (or uses default_df if none is
        passed).

        Args:
            df(DataFrame): pandas object for 'func' to be applied.
            func(function): to be applied to the DataFrame.
            **kwargs: any arguments to be passed to 'func'.
        """
        df = func(df, **kwargs)
        return self

    @column_list
    @check_df
    def auto_categorize(self, df = None, columns = None, threshold = 10):
        """Automatically assesses each column to determine if it has less than
        threshold unique values and is not boolean. If so, that column is
        converted to category type.

        Args:
            df(DataFrame): pandas object for columns to be evaluated for
                'categorical' type.
            columns(list): column names to be checked.
            threshold(int): number of unique values necessary to form a
                category. If there are less unique values than the threshold,
                the column is converted to a category type. Otherwise, it will
                remain its current datatype.

        Raises:
            KeyError: if a column in 'columns' is not in 'df'.
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                if not column in self.booleans:
                    if df[column].nunique() < threshold:
                        df[column] = df[column].astype('category')
                        self.datatypes[column] = 'categorical'
            else:
                error = column + ' is not in ingredients DataFrame'
                raise KeyError(error)
        return self

    @column_list
    @check_df
    def change_datatype(self, df = None, columns = None, datatype = None):
        """Changes column datatypes of columns passed or columns with the
        prefixes passed.

        The datatype becomes the new datatype for the columns in both the
        'datatypes' dict and in reality - a method is called to try to convert
        the column to the appropriate datatype.

        Args:
            df(DataFrame): pandas object for datatypes to be changed.
            columns(list): column names for datatypes to be changed.
            datatype(str): contains name of the datatype to convert the columns.
        """
        for column in columns:
            self.datatypes[column] = datatype
        self.convert_column_datatypes(df = df)
        return self

    @check_df
    def conform(self, df = None, step = None):
        """Adjusts some of the siMpLify-specific datatypes to the appropriate
        datatype based upon the current step.

        Args:
            df(DataFrame): pandas object for datatypes to be conformed.
            step(str): corresponding to the current state.
        """
        self.step = step
        for column, datatype in self.datatypes.items():
            if self.step in ['reap', 'clean']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.datatypes[column] = str
            elif self.step in ['bale', 'deliver']:
                if datatype in ['list', 'pattern']:
                    self.datatypes[column] = 'category'
        self.convert_column_datatypes(df = df)
        return self

    @check_df
    def convert_column_datatypes(self, df = None, raise_errors = False):
        """Attempts to convert all column data to the match the datatypes in
        'datatypes' dictionary.

        Args:
            df(DataFrame): pandas object with data to be changed to a new type.
            raise_errors(bool): whether errors should be raised when converting
            datatypes or ignored. Selecting False (the default) risks type
                mismatches between the datatypes listed in the 'datatypes' dict
                and 'df', but it prevents the program from being halted if
                an error is encountered.
        """
        if raise_errors:
            raise_errors = 'raise'
        else:
            raise_errors = 'ignore'
        for column, datatype in self.datatypes.items():
            if not isinstance(datatype, str):
                df[column].astype(dtype = datatype,
                                  copy = False,
                                  errors = raise_errors)
        # Attempts to downcast datatypes to simpler forms if possible.
        self.downcast(df = df)
        return self

    @column_list
    @check_df
    def convert_rare(self, df = None, columns = None, threshold = 0):
        """Converts categories rarely appearing within categorical columns
        to empty string if they appear below the passed threshold.

        The threshold is defined as the percentage of total rows.

        Args:
            df(DataFrame): pandas object with 'categorical' columns.
            columns(list): column names for datatypes to be checked. If it is
                not passed, all 'categorical' columns will be checked.
            threshold: a float indicating the percentage of values in rows
                below which a default value is substituted.

        Raises:
            KeyError: if column in 'columns' is not in 'df'.
        """
        if not columns:
            columns = self.categoricals
        for column in columns:
            if column in df.columns:
                df['value_freq'] = df[column].value_counts() / len(df[column])
                df[column] = np.where(df['value_freq'] <= threshold,
                                      self.default_values['categorical'],
                                      df[column])
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        if 'value_freq' in df.columns:
            df.drop('value_freq', axis = 'columns', inplace = True)
        return self

    @check_df
    def create_column_list(self, df = None, columns = None, prefixes = None,
                           mask = None):
        """Dynamically creates a new column list from a list of columns, lists
        of prefixes, and/or boolean mask.

        Args:
            df(DataFrame): pandas object.
            columns(list): column names to be included.
            prefixes(list): list of prefixes for columns to be included.
            mask(numpy array, list, or Series, of booleans): mask for columns to
                be included.

        Returns:
            column_names(list): column names created from 'columns', 'prefixes',
                and 'mask'.
        """
        column_names = []
        if (isinstance(mask, np.ndarray)
                or isinstance(mask, list)
                or isinstance(mask, pd.Series)):
            for boolean, feature in zip(mask, list(df.columns)):
                if boolean:
                    column_names.append(feature)
        else:
            temp_list = []
            prefixes_list = []
            if prefixes:
                for prefix in self.listify(prefixes):
                    temp_list = [col for col in df if col.startswith(prefix)]
                    prefixes_list.extend(temp_list)
            if columns:
                if prefixes:
                    column_names = self.listify(columns) + prefixes_list
                else:
                    column_names = self.listify(columns)
            else:
                column_names = prefixes_list
        return column_names

    @column_list
    def create_series(self, columns = None, return_series = True):
        """Creates a Series (row) with the 'datatypes' dict.

        Args:
            columns(list): index names for pandas Series.
            return_series (bool): whether the Series should be returned (True)
                or assigned to attribute named in 'default_df' (False).

        Returns:
            Either nothing, if 'return_series' is False or a pandas Series with
                index names matching 'datatypes' keys and datatypes matching
                'datatypes values'.
        """
        # If columns is not passed, the keys of self.datatypes are used.
        if not columns and self.datatypes:
            columns = list(self.datatypes.keys())
        row = pd.Series(index = columns)
        # Fills series with default_values based on datatype.
        if self.datatypes:
            for column, datatype in self.datatypes.items():
                row[column] = self.default_values[datatype]
        if return_series:
            return row
        else:
            setattr(self, self.default_df, row)
            return self

    @column_list
    @check_df
    def decorrelate(self, df = None, columns = None, threshold = 0.95):
        """Drops all but one column from highly correlated groups of columns.

        The threshold is based upon the .corr() method in pandas. columns can
        include any datatype accepted by .corr(). If columns is set to None,
        all columns in the DataFrame are tested.

        Args:
            df(DataFrame): pandas object to be have highly correlated features
                removed.
            threshold(float): the level of correlation using pandas corr method
            above which a column is dropped. The default threshold is 0.95,
                consistent with a common p-value threshold used in research.
        """
        if columns:
            corr_matrix = df[columns].corr().abs()
        else:
            corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                          k = 1).astype(np.bool))
        corrs = [col for col in upper.corrs if any(upper[col] > threshold)]
        self.drop_columns(columns = corrs)
        return self

    @column_list
    @check_df
    def downcast(self, df = None, columns = None, allow_unsigned = True):
        """Decreases memory usage by downcasting datatypes.

        For numerical datatypes, the method attempts to cast the data to
        unsigned integers if possible when 'allow_unsigned' is True. If more
        data might be added later which, in the same column, has values less
        than zero, 'allow_unsigned' should be set to False.

        Args:
            df(DataFrame): pandas object for columns to be downcasted.
            columns(list): columns to downcast.
            allow_unsigned(bool): whether to allow downcasting to unsigned int.

        Raises:
            KeyError: if column in 'columns' is not in 'df'.
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                if self.datatypes[column] in ['boolean']:
                    df[column] = df[column].astype(bool)
                elif self.datatypes[column] in ['integer', 'float']:
                    try:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'integer')
                        if min(df[column] >= 0) and allow_unsigned:
                            df[column] = pd.to_numeric(df[column],
                                                       downcast = 'unsigned')
                    except ValueError:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'float')
                elif self.datatypes[column] in ['categorical']:
                    df[column] = df[column].astype('category')
                elif self.datatypes[column] in ['list']:
                    df[column].apply(self.listify,
                                     axis = 'columns',
                                     inplace = True)
                elif self.datatypes[column] in ['datetime']:
                    df[column] = pd.to_datetime(df[column])
                elif self.datatypes[column] in ['timedelta']:
                    df[column] = pd.to_timedelta(df[column])
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    @column_list
    @check_df
    def drop_columns(self, df = None, columns = None):
        """Drops list of columns and columns with prefixes listed.

        In addition to removing the columns, any dropped columns have their
        column names stored in the cumulative 'dropped_columns' list. If you
        wish to make use of the 'dropped_columns' attribute, you should use this
        'drop_columns' method instead of dropping the columns directly.

        Args:
            df(DataFrame or Series): pandas object for columns to be dropped
            columns(list): columns to drop.
        """
        if isinstance(df, pd.DataFrame):
            df.drop(columns, axis = 'columns', inplace = True)
        else:
            df.drop(columns, inplace = True)
        self.dropped_columns.extend(columns)
        return self

    @column_list
    @check_df
    def drop_infrequent(self, df = None, columns = None, threshold = 0):
        """Drops boolean columns that rarely are True.

        This differs from the sklearn VarianceThreshold class because it is only
        concerned with rare instances of True and not False. This enables
        users to set a different variance threshold for rarely appearing
        information. threshold is defined as the percentage of total rows (and
        not the typical variance formulas used in sklearn).

        Args:
            df(DataFrame): pandas object for columns to checked for infrequent
                boolean True values.
            columns(list): columns to check.
            threshold(float): the percentage of True values in a boolean column
            that must exist for the column to be kept.
        """
        if not columns:
            columns = self.booleans
        infrequents = []
        for column in self.booleans:
            if column in columns:
                if df[column].mean() < threshold:
                    infrequents.append(column)
        self.drop_columns(columns = infrequents)
        return self

    def finalize(self):
        """Finalizes Ingredients class instance."""
        if self.verbose:
            print('Preparing ingredients')
        # If 'df' or other DataFrame attribute is a file path, the file located
        # there is imported.
        for df_name in self.options.keys():
            if (not(isinstance(getattr(self, df_name), pd.DataFrame) or
                    isinstance(getattr(self, df_name), pd.Series))
                    and getattr(self, df_name)
                    and os.path.isfile(getattr(self, df_name))):
                self.load(name = df_name, file_path = self.df)
        # If datatypes passed, checks to see if columns are in 'df'. Otherwise,
        # datatypes are inferred.
        self._initialize_datatypes()
        return self

    @check_df
    def infer_datatypes(self, df = None):
        """Infers column datatypes and adds those datatypes to types.

        This method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.) instead of
        simple types.

        This methods also allows the user to choose which datatypes to look for
        by changing the 'default_values' dict stored in 'all_datatypes'.

        Non-standard python datatypes cannot be inferred.

        Args:
            df(DataFrame): pandas object for datatypes to be inferred.
        """
        if not self.datatypes:
            self.datatypes = {}
        for datatype in self.all_datatypes.values():
            type_columns = df.select_dtypes(
                include = [datatype]).columns.to_list()
            self.datatypes.update(
                dict.fromkeys(type_columns,
                              self.all_datatypes[datatype]))
        return self

    def save_dropped(self, file_name = 'dropped_columns', file_format = 'csv'):
        """Saves 'dropped_columns' into a file

        Args:
            file_name(str): file name without extension of file to be exported.
            file_format(str): file format name.
        """
        # Deduplicates dropped_columns list
        self.dropped_columns = self.deduplicate(self.dropped_columns)
        if self.dropped_columns:
            if self.verbose:
                print('Exporting dropped feature list')
            self.depot.save(variable = self.dropped_columns,
                                folder = self.depot.experiment,
                                file_name = file_name,
                                file_format = file_format)
        elif self.verbose:
            print('No features were dropped during preprocessing.')
        return

    @column_list
    @check_df
    def smart_fill(self, df = None, columns = None):
        """Fills na values in a DataFrame with defaults based upon the datatype
        listed in the 'datatypes' dictionary.

        Args:
            df(DataFrame): pandas object for values to be filled
            columns(list): list of columns to fill missing values in. If no
                columns are passed, all columns are filled.

        Raises:
            KeyError: if column in 'columns' is not in 'df'.
        """
        for column in self._check_columns(columns):
            if column in df:
                default_value = self.all_datatypes.default_values[
                        self.datatypes[column]]
                df[column].fillna(default_value, inplace = True)
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    @check_df
    def split_xy(self, df = None, label = 'label'):
        """Splits df into x and y based upon the label ('y' column) passed.

        Args:
            df(DataFrame): initial pandas object to be split
            label(str or list): name of column(s) to be stored in self.y
        """
        self.x = df.drop(label, axis = 'columns')
        self.y = df[label]
        # Drops columns in self.y from datatypes dictionary and stores its
        # datatype in 'label_datatype'.
        self.label_datatype = {label : self.datatypes[label]}
        del self.datatypes[label]
        return self

    """ Properties """

    @property
    def full(self):
        """Returns the full dataset divided into x and y twice.

        This is used when the user wants the training and testing datasets to
        be the full dataset. This creates obvious data leakage problems, but
        is sometimes used after the model is tested and validated to produce
        metrics and results based upon all of the data."""
        return (self.options['x'], self.options['y'],
                self.options['x'], self.options['y'])

    @property
    def test(self):
        """Returns the test data."""
        return self.options['x_test'], self.options['y_test']

    @property
    def train(self):
        """Returns the training data."""
        return self.options['x_train'], self.options['y_train']

    @property
    def train_test(self):
        """Returns the training and testing data."""
        return (*self.train, *self.test)

    @property
    def train_test_val(self):
        """Returns the training, test, and validation data."""
        return (*self.train, *self.test, *self.val)

    @property
    def train_val(self):
        """Returns the training and validation data."""
        return (*self.train, *self.val)

    @property
    def val(self):
        """Returns the validation data."""
        return self.options['x_val'], self.options['y_val']

    @property
    def xy(self):
        """Returns the full dataset divided into x and y."""
        return self.options['x'], self.options['y']


@dataclass
class SimpleManager(SimpleClass, ABC):
    """Parent abstract base class for siMpLify planners, steps, and techniques.

    This class adds a required 'produce' method and other methods useful to
    siMpLify classes which transform data or fit models.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.
    """
    def __post_init__(self):
        super().__post_init__()
        # Outputs class status to console if verbose option is selected.
        # if self.verbose:
        #     print('Creating', self.name)
        return self

    """ Private Methods """

    def _create_option_lists(self):
        """Creates list of option lists of techniques."""
        self.all_steps = []
        for step in self.options.keys():
            # Stores each step attribute in a list
            if hasattr(self, step):
                setattr(self, step, self.listify(getattr(self, step)))
            else:
                setattr(self, step, ['none'])
            # Adds step to a list of all step lists
            self.all_steps.append(getattr(self, step))
        return self

    def _create_plan_instance(self, plan):
        plan_techniques = {}
        for j, (step, technique) in enumerate(self.options.items()):
            # Stores each step attribute in a dict.
            technique_instance = technique(technique = plan[j])
            plan_techniques.update({step : technique_instance})
        return self.plan_class(techniques = plan_techniques)

    def _create_plan_iterables(self):
        # Uses default name of 'plans' if class doesn't have 'plan_iterable'
        # attribute.
        if not hasattr(self, 'plan_iterable'):
            self.plan_iterable = 'plans'
            self.plans = {}
        # Creates empty dict for 'plan_iterable' attribute if attribute doesn't
        # exist.
        elif (not hasattr(self, self.plan_iterable)
                or getattr(self, self.plan_iterable) is None):
            setattr(self, self.plan_iterable, {})
        return self

    def _finalize_parallel(self):
        """Creates dict with steps as keys and techniques as values for each
        plan in the 'plans' attribute."""
        # Creates a list of all possible permutations of step techniques
        # selected. Each item in the the list is an instance of the plan class.
        self.all_plans = list(map(list, product(*self.all_steps)))
        # Iterates through possible steps and either assigns corresponding
        # technique from 'all_plans'.
        for i, plan in enumerate(self.all_plans):
            plan_instance = self._create_plan_instance(plan = plan)
            plan_instance.number = i
            getattr(self, self.plan_iterable).update({i : plan_instance})
        return self

    def _finalize_serial(self):
        """Creates dict with steps as keys and techniques as values."""
        self.all_plans = dict(zip(self.options.keys(), self.all_steps))
        # Changes 'plan_iterable' attribute to be instance of 'plan_class' with
        # the 'techniques' parameter being the prepared dict.
        setattr(self, self.plan_iterable, self.plan_class(
            techniques = self.all_plans))
        return self

    """ Public Methods """

    def draft(self):
        """ Declares defaults for class."""
        self.options = {}
        self.checks = ['steps', 'depot', 'ingredients']
        self.state_attributes = ['depot', 'ingredients']
        return self

    def finalize(self):
        """Finalizes"""
        self._create_option_lists()
        self._create_plan_iterables()
        getattr(self, '_finalize_' + self.manager_type)()
        return self

    @abstractmethod
    def produce(self, variable = None, **kwargs):
        """Required method that implements all of the finalized objects on the
        passed variable. The variable is returned after being transformed by
        called methods. It is roughly equivalent to the scikit-learn transform
        method.

        Args:
            variable: any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        pass
        return variable

    # Methods saved for possible later use.

    # def edit_parameters(self, step, parameters):
    #     """Adds parameter sets to the parameters dictionary of a prescribed
    #     step. """
    #     self.options[step].edit_parameters(parameters = parameters)
    #     return self

    # def edit_runtime_parameters(self, step, parameters):
    #     """Adds runtime_parameter sets to the parameters dictionary of a
    #     prescribed step."""
    #     self.options[step].edit_runtime_parameters(parameters = parameters)
    #     return self

    # def edit_step_class(self, step_name, step_class):
    #     self.options.update({step_name, step_class})
    #     return self

    # def edit_technique(self, step, technique, parameters = None):
    #     tool_instance = self.options[step](technique = technique,
    #                                        parameters = parameters)
    #     return tool_instance
