
from configparser import ConfigParser
from dataclasses import dataclass
import os
import re

from .cookbook import Countertop
from .cookbook import Recipe
from .cookbook.steps import Cleave
from .cookbook.steps import Custom
from .cookbook.steps import Encode
from .cookbook.steps import Mix
from .cookbook.steps import Model
from .cookbook.steps import Reduce
from .cookbook.steps import Sample
from .cookbook.steps import Scale
from .cookbook.steps import Split
from .critic import Presentation
from .critic import Review
from .inventory import Inventory


@dataclass
class Menu(Countertop):
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
        1) Pass file_path and the menu .ini file will automatically be loaded;
        2) Pass a prebuilt nested dictionary for storage in the Menu class;
            or
        3) The Menu class will automatically look for a file called
            simplify_menu.ini in the subdfolder 'menu' off of the
            working directory.

    Whichever option is chosen, the nested menu dictionary is stored in the
    attribute .config. Users can store any section of the config dictionary as
    attributes in a class instance by using the localize method.

    If set_types is set to True (the default option), the dictionary values are
    automatically converted to appropriate datatypes.

    If no_lists is set to True (the default is False), dictionaries containing
    ', ' will be returned as strings instead of lists.

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
                self.menu.localize()

    The result will be that an instance of Fakeclass will contain .verbose and
    .file_type as attributes that are appropriately typed.

    Because Menu uses ConfigParser, it only allows 2-level menu dictionaries.
    The desire for accessibility and simplicity dictated this limitation.

    Attributes:
        file_path: string of where the menu .ini file is located.
        config: two-level nested dictionary storing menu. If a file_path is
            provided, config will automatically be created.
        set_types: boolean variable determines whether values in config are
            converted to other types (True) or left as strings (False).
        no_list: if set_types = True, sets whether config values can be set
            to lists (False) if they contain a comma.
        auto_inject: if True, menu instance is automatically injected into
            appropriate siMpLify classes.
    """
    file_path : str = ''
    config : object = None
    set_types : bool = True
    no_lists : bool = False
    auto_inject : bool = True

    def __post_init__(self):
        """Initializes the config attribute and converts the dictionary values
        to the proper types if set_types is True.
        """
        self._set_config()
        self._set_types()
#        self._set_dependencies()
#        if self.auto_inject:
#            self.inject()
        return self

    def __delitem__(self, name):
        """Removes a dictionary section if name matches the name of a section.
        Otherwise, it will remove all entries with name inside the various
        sections of the config dictionary.
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
            error_message = name + ' not found in menu dictionary'
            raise KeyError(error_message)
        return self

    def __getitem__(self, value):
        """Returns a section of the config dictionary or, if none is found,
        an empty dictionary.
        """
        if value in self.config:
            return self.config[value]
        else:
            return {}

    def __setitem__(self, section, nested_dict):
        """Creates a new subsection in a specified section of the config
        nested dictionary.
        """
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

    def _listify(self, variable):
        """Checks to see if the methods are stored in a list. If not, the
        methods are converted to a list or a list of 'none' is created.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

    def _set_config(self):
        if not self.config:
            config = ConfigParser(dict_type = dict)
            config.optionxform = lambda option : option
            if not self.file_path:
                self.file_path = os.path.join('menu',
                                              'simplify_settings.ini')
            config.read(self.file_path)
            self.config = dict(config._sections)
        return self

    def _set_dependencies(self):
        """Sets nested config dictionaries to be injected into other classes
        in the siMpLify package.
        """
        self.dependencies = {Inventory : ['general', 'files'],
                             Recipe : ['general', 'recipes'],
                             Cleave : ['general', 'cleaver_parameters'],
                             Custom: ['general'],
                             Encode : ['general', 'encoder_parameters'],
                             Mix : ['general', 'mixer_parameters'],
                             Model : ['general', 'recipes'],
                             Reduce : ['general', 'reducer_parameters'],
                             Sample : ['general', 'sampler_parameters'],
                             Scale : ['general', 'scaler_parameters'],
                             Split : ['general', 'splitter_parameters'],
                             Presentation : ['general', 'review_parameters',
                                             'presentation_parameters'],
                             Review : ['general', 'recipes',
                                       'review_parameters']}
        return self

    def _set_types(self):
        """If set_types is True, all dictionary values in config are converted
        to the appropriate type.
        """
        if self.set_types:
            for section, nested_dict in self.config.items():
                for key, value in nested_dict.items():
                    self.config[section][key] = self._typify(value)
        return self

    def _typify(self, value):
        """Converts strings to list (if ', ' is present), int, float,
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

    def add_dependencies(self, dependencies, section_lists = None):
        """Adds dependencies for injection of appropriate settings.

        Parameters:
            dependencies: if a dictionary, it should be in the form of:
                {class : [sections]}
                if a list or str, it should include class(es) to be injected.
                section_lists should also be passed with corresponding list(s)
                of sections.
            section_lists: a str or list of the same length as dependencies
                which includes the section names containing attributes and
                values to be injected.
        """
        if isinstance(dependencies, dict):
            self.dependencies.update(dependencies)
        else:
            new_dependencies = zip(self._listify(dependencies),
                                   self._listify(section_lists))
            for dependency, section_list in new_dependencies.items():
                self.dependencies.update({dependency : section_list})
        self.inject()
        return self

    def add_settings(self, new_settings):
        """Adds a new settings to the config dictionary.

        Parameters:
           new_setting: can either be a dictionary or Menu object containing
               new attribute, value pairs or a string containing a file path
               from which new configuration options can be found."""
        if isinstance(new_settings, dict):
            self.config.update(new_settings)
        elif isinstance(new_settings, str):
            config = ConfigParser(dict_type = dict)
            config.optionxform = lambda option : option
            config.read(new_settings)
            self.config.update(dict(config._sections))
        elif isinstance(new_settings.config, dict):
            self.config.update(new_settings.config)
        else:
            error_message = 'new_options must be dict, Menu instance, or str'
            raise TypeError(error_message)
        return self

    def inject(self):
        """Injects appropriate settings from Menu.config into classes."""
        for dependency, sections in self.dependencies.items():
            for section in sections:
                if section in ['general', 'files', 'recipes',
                               'presentation_parameters',
                               'review_parameters']:
                    for key, value in self.config[section].items():
                        setattr(dependency, key, value)
                else:
                    setattr(dependency, 'parameters', self.config[section])
        Model.menu = self
        return self

    def localize(self, instance, sections, override = False):
        """Stores the section or sections of the config dictionary in the
        passed class instance as attributes to that class instance.
        """
        for section in self._listify(sections):
            for key, value in self.config[section].items():
                if not hasattr(instance, key) or override:
                    setattr(instance, key, value)
        return