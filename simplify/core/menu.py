
from configparser import ConfigParser
from dataclasses import dataclass
import re

from simplify.core.base import SimpleClass
from simplify.core.tools import listify


@dataclass
class Menu(SimpleClass):
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

        file_path: string of where the menu .ini file is located.
        configuration: two-level nested dictionary storing menu. If a file_path
            is provided, configuration will automatically be created.
        infer_types: boolean variable determines whether values in
            configuration are converted to other types (True) or left as
            strings (False).
        auto_prepare: sets whether to automatically call the prepare method
            when the class is instanced. If you do not plan to make any
            adjustments to the class otpions, this parameter should be set to
            True. If you plan to make such changes, prepare should be called
            when those changes are complete.
        auto_start: sets whether the start method should be called when the
            class is instanced. It should generally be set to true, unless
            auto_prepare is False and extensive changes to the menu options
            are anticipated.
    """
    file_path : str
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
        found, an empty dictionary.

        Parameters:

            key: the name of the dictionary key for which the value is
                sought.
        """
        if key in self.configuration:
            return self.configuration[key]
        else:
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

    def _create_configuration(self, file_path):
        """Creates a configuration dictionary from a file.

        Parameters:
            file_path: full path where the configuration file is located. The
                file format must be compatiable with ConfigParser.
        """
        configuration = ConfigParser(dict_type = dict)
        configuration.optionxform = lambda option : option
        configuration.read(file_path)
        configuration = dict(configuration._sections)
        return configuration

    def _infer_types(self):
        """If infer_types is True, all dictionary values in configuration are
        converted to the appropriate type.
        """
        if self.infer_types:
            for section, nested_dict in self.configuration.items():
                for key, value in nested_dict.items():
                    self.configuration[section][key] = self.typify(value)
        return self

    def _inject_base(self):
        """Injects parent class, SimpleClass with this Menu instance so that
        the instance is available to other files in the siMpLify package. It
        also adds the 'general' dictionary keys as attributes to SimpleClass.
        """
        SimpleClass.menu = self
        self.inject(instance = SimpleClass, sections = ['general'])
        return self

    def _set_defaults(self):
        """Loads configuration dictionary using ConfigParser if configuration
        does not presently exist.
        """
        if not self.configuration:
            self.configuration = self._create_configuration(
                    file_path = self.file_path)
        return self

    def add_settings(self, new_settings):
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
        for section in listify(sections):
            for key, value in self.configuration[section].items():
                if not hasattr(instance, key) or override:
                    setattr(instance, key, value)
        return

    def prepare(self):
        """Prepares instance of Menu."""
        self._infer_types()
        return self

    def start(self):
        """Injects Menu instance into base SimpleClass."""
        self._inject_base()
        return self

    @staticmethod
    def typify(variable):
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