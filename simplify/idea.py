"""
.. module:: idea
:synopsis: configures a siMpLify project
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from dataclasses import dataclass
from importlib import import_module
from itertools import product
import os
import re

from simplify.core.base import SimpleClass
from simplify.core.iterable import SimpleIterable


@dataclass
class Idea(SimpleClass):
    """Converts a data science idea into python.

    If configuration settings are imported from a file, Idea creates a nested
    dictionary, converting dictionary values to appropriate datatypes, and
    stores portions of the configuration dictionary as attributes in other
    classes. Idea is based on python's ConfigParser. It seeks to cure some
    of the shortcomings of the base ConfigParser getattr(self, name), including:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting items creates verbose code.
        3) It still OrderedDict (even though python 3.6+ has automatically
             orders regular dictionaries).

    To use the Idea class, the user can either pass to 'configuration':
        1) a file path, which will automatically be loaded into Idea;
        2) a file name which is located in the current working directory,
            which will automatically be loaded into Idea;
            or,
        3) a prebuilt nested dictionary matching the specifications of the
        'configuration' attribute.

    Whichever option is chosen, the nested Idea dictionary is stored in the
    attribute 'configuration'.

    If 'infer_types' is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes (str, list, float,
    bool, and int are currently supported)

    Users can add any key/value pairs in a section of the 'configuration'
    dictionary as attributes to a class instance by using the 'inject' method.

    For example, if the idea source file is as follows:

        [general]
        verbose = True
        seed = 43

        [files]
        source_format = csv
        test_data = True
        test_chunk = 500
        random_test_chunk = True

        [chef]
        chef_steps = split, reduce, model

    'verbose' and 'file_type' will automatically be added to every siMpLify
    class because they are located in the 'general' section. If a subclass
    wants attributes from the files section, then the following line should
    appear in __post_init__ before calling super().__post_init__:

        self.idea_sections = ['files']

    If the subclass wants the cookbook settings as well, then the code should
    be:
        self.idea_sections = ['files', 'chef']

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

    Idea will also automatically inject any sections of the 'configuration'
    as 'parameters' in a class instance if the section ends with '_parameters'
    and the prefix matches either the 'name' or 'technique' attribute.

    Args:
        configuration(str or dict): either a file path, file name, or two-level
            nested dictionary storing settings. If a file path is provided, a
            nested dict will automatically be created from the file and stored
            in 'configuration'. If a file name is provided, Idea will look for
            it in the current working directory and store its contents in
            'configuration'. If a dict is provided, it should be nested into
            sections with individual settings in key/value pairs.
        depot(Depot or str): an instance of Depot or a string containing the
            full path of where the root folder should be located for file
            output. Once a Depot instance is created, it is automatically made
            available to all other SimpleClass subclasses that are instanced in
            the future. If 'depot' is not passed, a default Depot instance will
            be created.
        ingredients(Ingredients or str): an instance of Ingredients, a string
            containing the full file path of where a data file for a pandas
            DataFrame is located, or a string containing a file name in the
            default data folder, as defined in the Depot instance.
        infer_types(bool): whether values in 'configuration' are converted to
            other types (True) or left as strings (False).
        name(str): as with other classes in siMpLify, the name is used for
            coordinating between classes. If Idea is subclassed, it is
            generally a good idea to keep the 'name' attribute as 'idea'.
        auto_publish(bool): whether to automatically call the 'publish'
            method when the class is instanced. Unless adding an additional
            source for 'configuration' settings, this should be set to True.
        auto_implement(bool): sets whether to automatically call the 'implement'
            method when the class is instanced.

    """
    configuration: object = None
    depot: object = None
    ingredients: object = None
    infer_types: bool = True
    name: str = 'idea'
    auto_publish: bool = True
    auto_implement: bool = False
    lazy_import: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __add__(self, other):
        self.update(new_settings = other)
        return self

    def __contains__(self, item):
        return item in self.configuration

    def __delitem__(self, key):
        """Removes a dictionary section if 'key' matches the name of a section.
        Otherwise, it will remove all entries with 'key' inside the various
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
            error = key + ' not found in idea dictionary'
            raise KeyError(error)
        return self

    def __getattr__(self, attr):
        """Intercepts dict method calls and applies them to 'configuration'.

        Args:
            attr (str): attribute sought.

        Returns:
            The matching dict method, an attribute, or None, if a matching
                attribute does not exist.

        Raises:
            AttributeError: if a dunder attribute is sought.
        """
        # Intecepts common dict methods and applies them to 'configuration'.
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
        """Returns a section of 'configuration' or key within a section.

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

    def __iadd__(self, other):
        self.update(new_settings = other)
        return self

    def __iter__(self):
        """Returns iterable configuration dict items()."""
        return self.configuration.items()

    def __len__(self):
        return len(self.configuration)

    def __radd__(self, other):
        self.update(new_settings = other)
        return self

    def __setitem__(self, section, dictionary):
        """Creates new key/value pair(s) in a specified section of
        'configuration'.

        Args:
            section(str): name of a section in 'configuration'.
            dictionary(dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'section' isn't a str or 'dictionary' isn't a dict.
        """
        if isinstance(section, str):
            if isinstance(dictionary, dict):
                if section in self.configuration:
                    self.configuration[section].update(dictionary)
                else:
                    self.configuration[section] = dictionary
            else:
                error = 'dictionary must be dict type'
                raise TypeError(error)
        else:
            error = 'section must be str type'
            raise TypeError(error)
        return self

    """ Private Methods """

    def _check_configuration(self):
        """Checks the datatype of 'configuration' and sets 'technique' to
        properly publish 'configuration'.

        Raises:
            AttributeError: if 'configuration' is None.
            TypeError: if 'configuration' is a path to a file that neither
                has an 'ini' nor 'py' extension or if 'configuration' is
                neither a str nor a dict.

        """
        if self.configuration:
            if isinstance(self.configuration, str):
                if self.configuration.endswith('.ini'):
                    self.technique = 'ini_file'
                elif self.configuration.endswith('.py'):
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
        """If 'infer_types' is True, values in 'configuration' are converted to
        the appropriate datatype.
        """
        if self.infer_types:
            for section, dictionary in self.configuration.items():
                for key, value in dictionary.items():
                    self.configuration[section][key] = self._typify(value)
        return self

    def _inject_base(self):
        """Injects parent class, SimpleClass, with this Idea so that it is
        available to other modules in the siMpLify getattr(self, name).
        """
        setattr(SimpleClass, 'idea', self)
        return self

    def _inject_parameters(self, instance, override):
        if instance.parameters is None or override:
            key_technique = instance.technique + '_parameters'
            key_name = instance.name + '_parameters'
            if key_technique in self.idea.configuration:
                instance.parameters = self.idea.configuration[key_technique]
            elif key_name in self.idea.configuration:
                instance.parameters = self.idea.configuration[key_name]
        return instance

    def _inject_plans(self, instance, override):
        """Creates cartesian product of all plans."""
        if hasattr(instance, 'comparer') and instance.comparer:
            plans = []
            for step in instance.sequence:
                key = step + '_techniques'
                if key in self.configuration[instance.name]:
                    plans.append(self.listify(self._convert_wildcards(
                            self.configuration[instance.name][key])))
                else:
                    plans.append(['none'])
            instance.plans = list(map(list, product(*plans)))
        return instance

    def _inject_sequence(self, instance, override):
         if not instance.sequence or override:
             instance.sequence = self.listify(instance._convert_wildcards(
                self.configuration[instance.name][instance.sequence_setting]))
         return instance

    def _load_from_ini(self, file_path = None):
        """Creates a configuration dictionary from an .ini file."""
        if file_path:
            configuration_file = file_path
        else:
            configuration_file = self.configuration
        if os.path.isfile(configuration_file):
            configuration = ConfigParser(dict_type = dict)
            configuration.optionxform = lambda option: option
            configuration.read(configuration_file)
            self.configuration = dict(configuration._sections)
        else:
            error = 'configuration file ' + configuration_file + ' not found'
            raise FileNotFoundError(error)
        return self

    def _load_from_py(self, file_path = None):
        """Creates a configuration dictionary from an .py file.

        Todo:
            file_path(str): path to python module with 'configuration' dict
                defined.

        """
        if file_path:
            configuration_file = file_path
        else:
            configuration_file = self.configuration
        if os.path.isfile(configuration_file):
            self.configuration = getattr(import_module(configuration_file),
                                         'configuration')
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

    def _typify(self, variable):
        """Converts stingsr to appropriate, supported datatypes.

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
        that key is stored at instance.steps (assuming that it does not exist
        or 'override' is True).

        If the sought key from a section has the '_techniques' suffix, the
        value for that key is stored either at the attribute named the prefix
        of the key (assuming that it does not exist or 'override' is True).

        Wildcard values of 'all', 'default', and 'none' are appropriately
        changed with the '_convert_wildcards' method.

        Args:
            instance(object): a class instance to which attributes should be
                added.
            sections(str or list(str)): the sections of 'configuration' which
                should be added to the instance.
            override(bool): if True, even existing attributes in instance will
                be replaced by 'configuration' key/value pairs. If False,
                current values in those similarly-named attributes will be
                maintained (unless they are None).

        Returns:
            instance with attribute(s) added.

        """
        for section in self.listify(sections):
            for key, value in self.configuration[section].items():
                setattr(instance, key, instance._convert_wildcards(value))
        # Injects appropriate 'parameters' into SimpleTechnique instance.
        if hasattr(instance, 'parameters'):
            instance = self._inject_parameters(
                    instance = instance,
                    override = override)
        if (instance.name in self.configuration
                and hasattr(instance, 'sequence_setting')):
            instance = self._inject_sequence(
                    instance = instance,
                    override = override)
            instance = self._inject_plans(
                    instance = instance,
                    override = override)
        return instance

    """ Core siMpLify Methods """

    def draft(self):
        """Sets options to create 'configuration' dictionary and checks to run
        on passed parameters."""
        super().draft()
        self.options = {
                'py_file': self._load_from_py,
                'ini_file': self._load_from_ini,
                'dict': None}
        self.checks.extend(['depot', 'ingredients'])
        return self

    def publish(self):
        """Prepares instance of Idea by checking passed configuration
        parameter and injecting Idea into SimpleClass.
        """
        self._check_configuration()
        if self.options[self.technique]:
            self.options[self.technique]()
        self._infer_types()
        self._inject_base()
        self.inject(instance = self, sections = ['general'])
        super().publish()
        return self

    def implement(self, ingredients = None):
        if ingredients:
            self.ingredients = ingredients
        self.simplify = Simplify()
        self.simplify.implement(ingredients = self.ingredients)
        return self

    """ Python Dictionary Compatibility Methods """

    def update(self, new_settings):
        """Adds new settings to the configuration dictionary.

        Args:
           new_settings(dict, str, or Idea): can either be a dicti or Idea
               object containing new key/value pairs, or a str containing a
               file path from which new configuration options can be found.

        Raises:
            TypeError: if 'new_settings' is neither a dict, str, or Idea
                instance.
        """
        if isinstance(new_settings, dict):
            self.configuration.update(new_settings)
        elif isinstance(new_settings, str):
            if new_settings.endswith('.ini'):
                technique = 'ini_file'
            elif new_settings.endswith('.py'):
                technique = self._load_py_file
            self.configuration.update(technique(file_path = new_settings))
        elif (hasattr(new_settings, 'configuration')
                and isinstance(new_settings.configuration, dict)):
            self.configuration.update(new_settings.configuration)
        else:
            error = 'new_options must be dict, Idea, or file path'
            raise TypeError(error)
        return self


@dataclass
class Simplify(SimpleIterable):
    """Controller class for siMpLify projects.

    This class is provided for applications that rely on Idea settings and/or
    subclass attributes. For a more customized application, users can access the
    subgetattr(self, name)s ('farmer', 'chef', 'critic', and 'artist') directly.

        name(str): name of class used to match settings sections in an Idea
            settings file and other portions of the siMpLify getattr(self, name). This is
            used instead of __class__.__name__ so that subclasses can maintain
            the same string name without altering the formal class name.
        auto_publish(bool): sets whether to automatically call the 'publish'
            method when the class is instanced. If you do not plan to make any
            adjustments beyond the Idea configuration, this option should be
            set to True. If you plan to make such changes, 'publish' should be
            called when those changes are complete.
        auto_implement(bool): sets whether to automatically call the 'implement'
            method when the class is instanced.

    """

    steps: object = None
    name: str = 'simplify'
    auto_publish: bool = True
    auto_implement: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    def __call__(self, ingredients = None):
        """Calls the class as a function.

        Args:

            ingredients(Ingredients or str): an instance of Ingredients, a
                string containing the full file path of where a data file for a
                pandas DataFrame is located, or a string containing a file name
                in the default data folder, as defined in a Depot instance.

        """
        self.__post_init__()
        self.implement(ingredients = ingredients)
        return self

    def _implement_dangerous(self):
        first_step = True
        for name in self.sequence:
            if first_step:
                first_step = False
                getattr(self, name).implement(ingredients = self.ingredients)
            else:
                getattr(self, name).implement(previous_step = previous_step)
            previous_step = getattr(self, name)
        return self

    def _implement_safe(self):
        for name in self.sequence:
            if name in ['farmer']:
                getattr(self, name).implement(ingredients = self.ingredients)
                self.ingredients = getattr(self, name).ingredients
                delattr(self, name)
            if name in ['chef']:
                getattr(self, name).implement(ingredients = self.ingredients)
                self.ingredients = getattr(self, name).ingredients
                self.recipes = getattr(self, name).recipes
                delattr(self, name)
            if name in ['critic', 'artist']:
                getattr(self, name).implement(
                    recipes = self.recipes)
                self.recipes = getattr(self, name).recipes
                delattr(self, name)
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
                'farmer': ['simplify.farmer', 'Almanac'],
                'chef': ['simplify.chef', 'Cookbook'],
                'critic': ['simplify.critic', 'Review'],
                'artist': ['simplify.artist', 'Canvas']}
        self.sequence_setting = 'packages'
        return self

    def implement(self, ingredients = None):
        if ingredients:
            self.ingredients = ingredients
        if self.conserve_memory:
            self._implement_safe()
        else:
            self._implement_dangerous()
        return self