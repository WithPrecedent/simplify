"""
Class for loading a settings file, creating a nested dictionary, converting
dictionary values to appropriate data types, enabling nested dictionary
lookups by user, and storing portions of the configuration dictionary as
attributes in other classes.

The Settings class is largely a wrapper for python's ConfigParser. It seeks
to cure some of the most significant shortcomings of the base ConfigParser
package:
    1) All values in ConfigParser are strings by default.
    2) The nested structure for getting items creates verbose code.
    3) It still uses OrderedDict (even though python 3.6+ has automatically
         orders regular dictionaries).

To use the Settings class, the user can either:
    1) Pass file_path and the settings file will automatically be loaded;
    2) Pass a prebuilt nested dictionary for storage in the Settings class; or
    3) The Settings class will automatically looking for a file called
        settings.ini in the subdfolder 'settings' off of the working
        directory.

Whichever option is chosen, the nested settings dictionary is stored in the
attribute .config. Users can store any section of the config dictionary as
attributes in a class instance by using the localize method.

If set_types is set to True (the default option), the dictionary values are
automatically converted to appropriate types.

If no_lists is set to True (the default is False), dictionaries containing
', ' will be returned as strings instead of lists.

For example, if the settings file (settings.ini stored in the appropriate
folder) is as follows:

[general]
verbose = True
file_type = csv

[files]
file_name = 'test_file'
iterations = 4

This code will create the settings file and store the general section as local
attributes in the class:

    class FakeClass(object):

        def __init__(self):
            self.settings = Settings()
            self.settings.localize()

The result will be that an instance of Fakeclass will contain .verbose and
.file_type as attributes that are appropriately typed.

Because Settings uses ConfigParser, it only allows 2-level settings
dictionaries. The desire for accessibility and simplicity dictated this
limitation.
"""

from configparser import ConfigParser
from dataclasses import dataclass
import os
import re


@dataclass
class Settings(object):

    file_path : str = ''
    config : object = None
    set_types : bool = True
    no_lists : bool = False

    def __post_init__(self):
        if not self.config:
            config = ConfigParser(dict_type = dict)
            config.optionxform = lambda option : option
            if not self.file_path:
                self.file_path = os.path.join('settings',
                                              'simplify_settings.ini')
            config.read(self.file_path)
            self.config = dict(config._sections)
        if self.set_types:
            for section, nested_dict in self.config.items():
                for key, value in nested_dict.items():
                    self.config[section][key] = self._typify(value)
        return self

    def __getitem__(self, value):
        if value in self.config:
            return self.config[value]
        else:
            error_message = value + ' not found in settings dictionary'
            raise KeyError(error_message)
            return

    def __setitem__(self, section, nested_dict):
        if isinstance(section, str):
            if isinstance(nested_dict, dict):
                self.config.update({section, nested_dict})
            else:
                error_message = 'nested_dict must be dict type'
                raise TypeError(error_message)
        else:
            error_message = 'section must be str type'
            raise TypeError(error_message)
        return self

    def __delitem__(self, name):
        """
        Magic method will remove a dictionary section if name matches the name
        of a section. Otherwise, it will remove all entries with name inside
        the various sections of the config dictionary.
        """
        found_value = False
        if name in self.config:
            found_value = True
            self.config.pop(name)
        else:
            for key, value in self.config.items():
                if name in value:
                    found_value = True
                    self.config[key].pop(name)
        if not found_value:
            error_message = name + ' not found in settings dictionary'
            raise KeyError(error_message)
        return self

    def _listify(self, variable):
        """
        Checks to see if the variable is currently a list type. If the variable
        is None, it is converted to a list with the string 'none'. If it is a
        string, it is converted to a list with that string. If the variable
        is already a list, it is returned unchanged.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

    def _typify(self, value):
        """
        Method that converts strings to list (if ', ' is present), int, float,
        or boolean types based upon the content of the string imported from
        ConfigParser.
        """
        if ', ' in value and not self.no_lists:
            return value.split(', ')
        elif re.search('\d', value):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        elif value in ['True', 'true', 'TRUE']:
            return True
        elif value in ['False', 'false', 'FALSE']:
            return False
        elif value in ['None', 'none', 'NONE']:
            return None
        else:
            return value

    def localize(self, instance, sections):
        """
        Stores the section or sections of the config dictionary in the passed
        class instance as attributes to that class instance.
        """
        for section in self._listify(sections):
            for key, value in self.config[section].items():
                setattr(instance, key, value)
        return

    def update(self, new_settings):
        """
        Adds a new nested dictionary to the config dictionary.
        """
        if (isinstance(new_settings, dict)
                or isinstance(new_settings.config, dict)):
            self.config.update(new_settings.config)
        else:
            error_message = 'new_settings must be a 2-level nested dict'
            raise TypeError(error_message)
        return self