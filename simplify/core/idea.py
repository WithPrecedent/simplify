"""


Contents:
    
    Idea: class which stores settings used throughout the siMpLify package.
        Idea is automatically injected into SimpleClass to allow all of its
        subclasses to access the Idea instance through the attribute 'idea'.
"""
from configparser import ConfigParser
from dataclasses import dataclass
import glob
import os
import pickle

from simplify.core.container import SimpleContainer


@dataclass
class Idea(SimpleContainer):
    """Loads and/or stores the user's idea, in the form of settings.

    Idea creates a nested dictionary, converting dictionary values to 
    appropriate datatypes, enabling nested dictionary lookups by user, and
    storing portions of the configuration dictionary as attributes in other
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
        3) Pass a prebuilt nested dictionary for storage in the Idea class.

    Whichever option is chosen, the nested idea dictionary is stored in the
    attribute .configuration. Users can store any key/value pairs in a section
    of the configuration dictionary as attributes in a class instance by using
    the inject method.

    If infer_types is set to True (the default option), the dictionary values
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

    verbose and file_type will automatically be added to every siMpLify class
    because they are located in the 'general' section. If a subclass wants
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
        
        
    Because Idea uses ConfigParser, it only allows 2-level idea dictionaries.
    The desire for accessibility and simplicity dictated this limitation.

    Parameters:
        configuration: either a file path, file name, or two-level nested
            dictionary storing settings. If a file path is provided, A nested
            dict will automatically be created from the file and stored in
            'configuration'. If a file name is provided, Idea will look for it
            in the current working directory and store its contents in
            'configuration'.
        infer_types: boolean variable determines whether values in
            'configuration' are converted to other types (True) or left as
            strings (False).
        auto_finalize: sets whether to automatically call the 'finalize' method
            when the class is instanced. Unless adding a new source for
            configuration settings, this should be set to True.
        auto_produce: sets whether to automatically call the 'produce' method
            when the class is instanced. Unless adding a new source for
            configuration settings, this should be set to True.
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
            error_message = key + ' not found in idea dictionary'
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

    def __iter__(self):
        """Returns iterable configuration dict items()."""
        return self.items()

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

    """ Private Methods """
    
    def _check_configuration(self):
        """Checks the datatype of 'configuration' and sets 'technique' to
        properly finalize 'configuration'.
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
        configuration = ConfigParser(dict_type = dict)
        configuration.optionxform = lambda option : option
        configuration.read(self.configuration)
        self.configuration = dict(configuration._sections)
        return self

    def _create_from_py(self):
        """Creates a configuration dictionary from an .py file."""
        pass
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
        """Injects parent class, SimpleClass with this Idea instance so that
        the instance is available to other files in the siMpLify package. It
        also adds the 'general' dictionary keys as attributes to SimpleClass.
        This ensures that every subclass of SimpleClass will have direct access
        to key, value pairs in the 'general' section of Idea as local 
        attributes.
        """
        SimpleClass.idea = self
        self.inject(instance = SimpleClass, sections = ['general'])
        return self

    def _typify(self, variable):
        """Converts strings to list (if ', ' is present), int, float, or
        boolean datatypes based upon the content of the string. If no
        alternative datatype is found, the variable is returned in its original
        form.
        
        Parameters:
            variable: string to be converted to appropriate datatype.
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
        """Loads configuration dictionary using ConfigParser if configuration
        does not presently exist.
        """
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
        """Stores the section or sections of the configuration dictionary in
        the passed class instance as attributes to that class instance.

        Parameters:
            instance: either a class instance or class to which attributes
                should be added.
            sections: the sections of the configuration dictionary which should
                have key, value pairs added as attributes to instance.
            override: if True, even existing attributes in instance will be
                replaced by configuration dictionary items. If False, current
                values in those similarly-named attributes will be maintained.
        """
        for section in self.listify(sections):
            for key, value in self.configuration[section].items():
                if not hasattr(instance, key) or override:
                    setattr(instance, key, value)
        return

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

        Parameters:
           new_settings: can either be a dictionary or Idea object containing
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
            error_message = 'new_options must be dict, Idea instance, or path'
            raise TypeError(error_message)
        return self
      
    def values(self):
        """Returns values (sections) of 'configuration' dict to mirror dict
        functionality.
        """
        return self.configuration.values()