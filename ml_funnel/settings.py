"""
Class for loading a settings file, creating a nested dictionary, and enabling
lookups by user.
"""

from configparser import ConfigParser
from dataclasses import dataclass
import os
import re

@dataclass
class Settings(object):

    file_path : str = ''
    settings_dict : object = None
    set_types : bool = True
    no_lists : bool = False

    def __post_init__(self):
        if not self.settings_dict:
            config = ConfigParser(dict_type = dict)
            config.optionxform = lambda option : option
            if not self.file_path:
                self.file_path = os.path.join('..', 'settings', 'settings.ini')
            config.read(self.file_path)
            self.config = dict(config._sections)
        else:
            self.config = self.settings_dict
        if self.set_types and (self.settings_dict or self.file_path):
            for section, nested_dict in self.config.items():
                for key, value in nested_dict.items():
                    self.config[section][key] = self._typify(value)
        return self

    def __getitem__(self, section, nested_key = None):
        if nested_key:
            return self.config[section][nested_key]
        else:
            return self.config[section]

    def __setitem__(self, section, nested_dict):
        self.config.update({section, nested_dict})
        return self

    def _typify(self, value):
        """
        Method that converts strings to list (if comma present), int, float,
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

    def simplify(self, class_instance, sections):
        if not isinstance(sections, list):
            sections = [sections]
        for section in sections:
            for key, value in self.config[section].items():
                setattr(class_instance, key, value)
        return

    def update(self, new_settings):
        self.config.update(new_settings.config)
        return self