"""
.. module:: idea
:synopsis: stores settings and configuration for siMpLify project
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from dataclasses import dataclass
import os
import re

from simplify.core.base import SimpleClass


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
    of the 'configuration' dictionary as attributes in a class instance by
    using the 'inject' method.

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
    class because they are located in the 'general' section. If a subclass
    wants attributes from the files section, then the following line should
    appear in __post_init__ before calling super().__post_init__:

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

    Regardless of the idea_sections added, all Idea settings can be similarly
    accessed using dict keys or local attributes. For example:

        self.idea['general']['seed'] # typical dict access technique

        self.idea['seed'] # if no section or other key is named 'seed'

        self.seed # exists because 'seed' is in the 'general' section

                            all return 43.

    Because Idea uses ConfigParser, it only allows 2-level dictionaries. The
    desire for accessibility and simplicity dictated this limitation.

    Args:
        configuration(str or dict): either a file path, file name, or two-level
            nested dictionary storing settings. If a file path is provided, a
            nested dict will automatically be created from the file and stored
            in 'configuration'. If a file name is provided, Idea will look for
            it in the current working directory and store its contents in
            'configuration'.
        infer_types(bool): whether values in 'configuration' are converted to
            other types (True) or left as strings (False).
        auto_publish(bool): whether to automatically call the 'publish'
            method when the class is instanced. Unless adding a new source for
            'configuration' settings, this should be set to True.
    """
    configuration: object = None
    infer_types: bool = True
    auto_publish: bool = True

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

    def __getattr__(self, attr):
        """Returns dict methods applied to configuration attribute if those
        methods are sought from the class instance.

        Args:
            attr (str): attribute sought.

        Returns:
            attribute or None, if attribute does not exist.

        Raises:
            AttributeError: if a dunder attribute is sought.
        """
        # Intecepts common dict methods and applies them to 'configuration'
        # dict.
        if attr in ['clear', 'items', 'pop', 'keys', 'values']:
            return getattr(self.configuration, attr)
        elif attr in self.__dict__:
            return self.__dict__[attr]
        elif attr.startswith('__') and attr.endswith('__'):
            error = 'Access to magic methods not permitted through __getattr__'
            raise AttributeError(error)
        else:
            error = attr + ' not found in ' + self.__class__.__name__
            raise AttributeError(error)

    def __getitem__(self, key):
        """Returns a section of the configuration or key within a section.

        Args:
            key(str): the name of the dictionary key for which the value is
                sought.

        Returns:
            dict if 'key' matches a section in 'configuration'. If 'key'
                matches a key within a section, the value, which can be any of
                the supported datatypes is returned. If no match is found an
                empty dict is returned.
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
        return self.configuration.items()

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
        properly publish 'configuration'.

        Raises:
            AttributeError if 'configuration' attribute is neither a file path
                dict, nor Idea instance.
            TypeError if 'configuration' is a string path to a file that neither
                has an 'ini' nor 'py' extension or if 'configuration' is neither
                a string nor a dictionary.
            FileNotFoundError if 'configuration' is a string path to a file that
                does not exist.
        """
        if self.configuration:
            if isinstance(self.configuration, str):
                if '.ini' in self.configuration:
                    self.technique = 'ini_file'
                elif '.py' in self.configuration:
                    self.technique = 'py_file'
                else:
                    error = 'configuration file must be .py or .ini file'
                    raise TypeError(error)
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
        """Injects parent class, SimpleClass with this Idea so that it is
        available to other modules in the siMpLify package.
        """
        setattr(SimpleClass, 'idea', self)
        setattr(SimpleClass, 'step', self.step)
        return self

    def _load_from_ini(self):
        """Creates a configuration dictionary from an .ini file."""
        if os.path.isfile(self.configuration):
            configuration = ConfigParser(dict_type = dict)
            configuration.optionxform = lambda option: option
            configuration.read(self.configuration)
            self.configuration = dict(configuration._sections)
        else:
            error = 'configuration file ' + self.configuration + ' not found'
            raise FileNotFoundError(error)
        return self

    def _load_from_py(self):
        """Creates a configuration dictionary from an .py file.

        Todo:
            Add .py file implementation.
        """
        pass
        return self

    @staticmethod
    def _numify(variable):
        """Attempts to convert 'variable' to a numeric type.

        Args:
            variable(str): variable to be converted.

        Returns
            variable(int, float, str) converted to numeric type, if possible.
        """
        try:
            return int(variable)
        except ValueError:
            try:
                return float(variable)
            except ValueError:
                return variable

    def _set_initial_state(self):
        """Sets initial 'step' for state_depenent subclasses."""
        if 'farmer' in self.configuration['general']['packages']:
            self.step = self.listify(
                    self.idea.configuration['almanac']['almanac_steps'])[0]
        elif 'chef' in self.configuration['general']['packages']:
            self.step = 'cook'
        elif 'review' in self.configuration['general']['packages']:
            self.step = 'review'
        else:
            self.step = 'canvas'
        return self

    def _typify(self, variable):
        """Converts str to appropriate, supported datatype.

        The method converts strings to list (if ', ' is present), int, float,
        or bool datatypes based upon the content of the string. If no
        alternative datatype is found, the variable is returned in its original
        form.

        Args:
            variable(str): string to be converted to appropriate datatype.

        Returns:
            variable(str, list, int, float, or bool): converted variable.
        """
        if ', ' in variable:
            variable = variable.split(', ')
            return [self._numify(v) for v in variable]
        elif re.search('\d', variable):
            return self._numify(variable)
        elif variable in ['True', 'true', 'TRUE']:
            return True
        elif variable in ['False', 'false', 'FALSE']:
            return False
        elif variable in ['None', 'none', 'NONE']:
            return None
        else:
            return variable

    """ Public Tool Methods """

    def inject(self, instance, sections, override = False):
        """Stores the section or sections of the 'configuration' dictionary in
        the passed class instance as attributes to that class instance. 
        
        If the sought section has the '_parameters' suffix, the section is 
        returned as a single dictionary at instance.parameters (assuming that 
        it does not exist or 'override' is True).
        
        If the sought key from a section has the '_steps' suffix, the value for
        that key is stored at instance.steps (assuming that it does not exist or
        'override' is True).
        
        If the sought key from a section has the '_techniques' suffix, the value 
        for that key is stored either at the attribute named the prefix of the 
        key (assuming that it does not exist or 'override' is True).
        
        Wildcard values of 'all', 'default', and 'none' are appropriately 
        changed with the '_convert_wildcards' method.

        Args:
            instance(object): either a class instance or class to which
                attributes should be added.
            sections(str or list(str)): the sections of the configuration 
                dictionary which should be added to the instance.
            override (bool): if True, even existing attributes in instance will
                be replaced by configuration dictionary items. If False,
                current values in those similarly-named attributes will be
                maintained (unless they are None).

        Returns:
            instance with attribute(s) added.
            
        """
        for section in self.listify(sections):
            if (section.endswith('_parameters') 
                    and (not instance.exists('parameters') or override)):
                instance.parameters = self.configuration[section]
            else:
                for key, value in self.configuration[section].items():
                    if (key.endswith('_steps') 
                            and (not instance.exists('steps') or override)):
                        instance.steps = instance._convert_wildcards(value)
                    elif key.endswith('_technique'):
                        attribute_name = key.replace('_technique', '')
                        if not instance.exists(attribute_name) or override:
                            setattr(instance, attribute_name, 
                                    instance._convert_wildcards(value))    
                    elif not instance.exists(key) or override:
                        setattr(instance, key, 
                                instance._convert_wildcards(value))
        return instance

    """ Core siMpLify Methods """

    def draft(self):
        """Sets options to create 'configuration' dict'."""
        # Sets options for creating 'configuration'.
        self.options = {'py_file': self._load_from_py,
                        'ini_file': self._load_from_ini,
                        'dict': None}
        return self

    def publish(self):
        """Prepares instance of Idea by checking passed configuration
        parameter and injecting Idea into SimpleClass.
        """
        self._check_configuration()
        if self.options[self.technique]:
            self.options[self.technique]()
        self._infer_types()
        # Sets 'step' to first step from Idea instance.
        self._set_initial_state()
        self._inject_base()
        return self

    """ Python Dictionary Compatibility Methods """

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
            error_message = 'new_options must be dict, Idea , or file path'
            raise TypeError(error_message)
        return self