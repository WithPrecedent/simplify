
from configparser import ConfigParser
from dataclasses import dataclass
import os
import re

from simplify.core.base import SimpleClass
from simplify.core.tools import SimpleUtilities


@dataclass
class Menu(SimpleClass, SimpleUtilities):
    """Loads and/or stores user settings.

    Menu creates a nested dictionary, converting dictionary values to
    appropriate data types, enabling nested dictionary lookups by user, and
    storing portions of the configuration dictionary as attributes in other
    classes. Menu is largely a wrapper for python's ConfigParser. It seeks
    to cure some of the most significant shortcomings of the base ConfigParser
    package:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting items creates verbose code.
        3) It still uses OrderedDict (even though python 3.6+ has automatically
             orders regular dictionaries).

    To use the Menu class, the user can either:
        1) Pass file_path and the menu .ini file will automatically be loaded,
            or;
        2) Pass a prebuilt nested dictionary for storage in the Menu class.

    Whichever option is chosen, the nested menu dictionary is stored in the
    attribute .configuration. Users can store any key/value pairs in a section
    of the configuration dictionary as attributes in a class instance by using
    the inject method.

    If infer_types is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes.

    For example, if the menu file (simplify_menu.ini stored in the appropriate
    folder) is as follows:

    [general]
    verbose = True
    file_type = csv

    [files]
    file_name = 'test_file'
    iterations = 4

    This code will create the menu file and store the general section as
    local attributes in the class:

        class FakeClass(object):

            def __init__(self):
                self.menu = Menu()
                self.menu.inject(instance = self, sections = ['general'])

    The result will be that an instance of Fakeclass will contain verbose and
    file_type as attributes that are appropriately typed.

    Because Menu uses ConfigParser, it only allows 2-level menu dictionaries.
    The desire for accessibility and simplicity dictated this limitation.

    Parameters:
        configuration: either a file_path or two-level nested dictionary storing
            settings. If a file_path is provided, A nested dict will
            automatically be created from the file and stored in
            'configuration'.
        infer_types: boolean variable determines whether values in
            'configuration' are converted to other types (True) or left as
            strings (False).
        auto_prepare: sets whether to automatically call the 'prepare' method
            when the class is instanced. Unless adding a new source for
            configuration settings, this should be set to True.
        auto_start: sets whether to automatically call the 'start' method
            when the class is instanced. Unless adding a new source for
            configuration settings, this should be set to True.
    """
    configuration : object = None
    infer_types : bool = True
    auto_prepare : bool = True
    auto_start : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def __delitem__(self, key):
        """Removes a dictionary section if name matches the name of a section.
        Otherwise, it will remove all entries with name inside the various
        sections of the configuration dictionary.

        Parameters:

            key: the name of the dictionary key or section to be deleted.
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
            error_message = key + ' not found in menu dictionary'
            raise KeyError(error_message)
        return self

    def __getitem__(self, key):
        """Returns a section of the configuration dictionary or, if none is
        found, it looks at keys within sections. If no match is still found,
        it returns an empty dictionary.

        Parameters:

            key: the name of the dictionary key for which the value is
                sought.
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


    def __setitem__(self, section, nested_dict):
        """Creates a new subsection in a specified section of the configuration
        nested dictionary.

        Parameters:

            section: a string naming the section of the configuration
                dictionary.
            nested_dict: the dictionary to be placed in that section.
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

    def __str__(self):
        """Returns the configuration dictionary."""
        return self.configuration

    def _check_configuration(self):
        if self.configuration:
            print(os.path.abspath(self.configuration))
            if os.path.isfile(os.path.abspath(self.configuration)):
                if '.ini' in self.configuration:
                    self.technique = 'ini_file'
                elif '.py' in self.configuration:
                    self.technique = 'py_file'
                else:
                    error = 'configuration file must be .py or .ini file'
                    raise FileNotFoundError(error)
            elif not isinstance(self.configuration, dict):
                error = 'configuration must be dict or file path'
                raise TypeError(error)
        else:
            error = 'configuration dict or path needed to instance Menu'
            raise AttributeError(error)
        return self

    def _create_from_ini(self):
        """Creates a configuration dictionary from an .ini file."""
        configuration = ConfigParser(dict_type = dict)
        configuration.optionxform = lambda option : option
        configuration.read(self.configuration)
        self.configuration = dict(configuration._sections)
        return self

    def _create_from_py(self):
        """Creates a configuration dictionary from an .py file."""
        pass
        return self

    def _define(self):
        """Loads configuration dictionary using ConfigParser if configuration
        does not presently exist.
        """
        # Lists tools from siMpLify package that should be added as local
        # staticmethods.
        self.tools = ['listify', 'typify']
        # Sets options for creating 'configuration'.
        self.options = {'py_file' : self._create_from_py,
                        'ini_file' : self._create_from_ini,
                        'dict' : None}
        return self

    def _infer_types(self):
        """If infer_types is True, all dictionary values in configuration are
        converted to the appropriate type.
        """
        if self.infer_types:
            for section, nested_dict in self.configuration.items():
                for key, value in nested_dict.items():
                    self.configuration[section][key] = self._typify(value)
        return self

    def _inject_base(self):
        """Injects parent class, SimpleClass with this Menu instance so that
        the instance is available to other files in the siMpLify package. It
        also adds the 'general' dictionary keys as attributes to SimpleClass.
        """
        SimpleClass.menu = self
        self.inject(instance = SimpleClass, sections = ['general'])
        return self

    def _typify(self, variable):
        """Converts strings to list (if ', ' is present), int, float, or
        boolean datatypes based upon the content of the string. If no
        alternative datatype is found, the variable is returned in its original
        form.
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

    def inject(self, instance, sections, override = False):
        """Stores the section or sections of the configuration dictionary in
        the passed class instance as attributes to that class instance.

        Parameters:

            instance: either a class instance or class to which attributes
                should be added.
            sections: the sections of the configuration dictionary which should
                have items added as attributes to instance.
            override: if True, even existing attributes in instance will be
                replaced by configuration dictionary items.
        """
        for section in self.listify(sections):
            for key, value in self.configuration[section].items():
                if not hasattr(instance, key) or override:
                    setattr(instance, key, value)
        return

    def prepare(self):
        """Prepares instance of Menu by checking passed configuration parameter.
        """
        self._check_configuration()
        return self

    def start(self):
        """Creates configuration setttings and injects Menu into SimpleClass.
        """
        if self.options[self.technique]:
            self.options[self.technique]()
        self._infer_types()
        self._inject_base()
        return self

    def update(self, new_settings):
        """Adds new settings to the configuration dictionary.

        Parameters:

           new_settings: can either be a dictionary or Menu object containing
               new attribute, value pairs or a string containing a file path
               from which new configuration options can be found.
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
            error_message = 'new_options must be dict, Menu instance, or path'
            raise TypeError(error_message)
        return self