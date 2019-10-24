"""
.. module:: idea
:synopsis: converts an idea into a siMpLify project
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from dataclasses import dataclass
from importlib import import_module
import os
import re

from simplify.core.base import SimpleClass
from simplify.core.controller import Simplify


@dataclass
class Idea(SimpleClass):
    """Converts a data science idea into python.

    If 'options' are imported from a file, Idea creates a nested dictionary,
    converting dictionary values to appropriate datatypes, and stores portions
    of the 'options' dictionary as attributes in other classes. Idea is based
    on python's ConfigParser. It seeks to cure some of the shortcomings of the
    base ConfigParser getattr(self, name), including:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting items creates verbose code.
        3) It uses OrderedDict (python 3.6+ orders regular dictionaries).

    To use the Idea class, the user can either pass to 'options':
        1) a file path, which will automatically be loaded into Idea;
        2) a file name which is located in the current working directory,
            which will automatically be loaded into Idea;
                                or,
        3) a prebuilt nested dictionary matching the specifications of the
        'options' attribute.

    If 'infer_types' is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes (str, list, float,
    bool, and int are currently supported)

    Whichever option is chosen, the nested Idea dictionary is stored in the
    attribute 'options'. However, dictionary access methods can either be
    applied to the 'options' dictionary (e.g., idea.options['general']) or an
    Idea instance (e.g., idea['general']). If using the dictionary 'update'
    method, it is better to apply it to the Idea instance because the Idea
    method is more flexible in handling different kinds of arguments.

    Users can add any key/value pairs from a section of the 'options'
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
        chef_techniques = split, reduce, model

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
                self.chef_techniques = ['split', 'reduce', 'model']
                return self

    Regardless of the idea_sections added, all Idea settings can be similarly
    accessed using dict keys or local attributes. For example:

        self.idea['general']['seed'] # typical dict access technique

        self.idea['seed'] # if no section or other key is named 'seed'

        self.seed # exists because 'seed' is in the 'general' section

                            all return 43.

    Within the siMpLify ecosystem, settings of two types take on particular
    importance and are automatically injected to certain classes:
        'parameters': settings with the suffix '_parameters' are automatically
            added to classes where the prefix matches the class's 'name' or
            'technique'.
        'techniques': for subclasses of 'SimplePackage', settings with the
            suffix '_techniques' are automatically added to classes with the
            prefix as the name of the attribute. These techniques are stored
            in lists used to create the permutations and/or sequences of
            possible steps in subclasses of SimplePlan.

    Because Idea uses ConfigParser, it only allows 2-level dictionaries. The
    desire for accessibility and simplicity dictated this limitation.

    Args:
        name (str): as with other classes in siMpLify, the name is used for
            coordinating between classes. If Idea is subclassed, it is
            generally a good idea to keep the 'name' attribute as 'idea'.
        options (str or dict): either a file path, file name, or two-level
            nested dictionary storing settings. If a file path is provided, a
            nested dict will automatically be created from the file and stored
            in 'options'. If a file name is provided, Idea will look for
            it in the current working directory and store its contents in
            'options'. If a dict is provided, it should be nested into
            sections with individual settings in key/value pairs.
        depot (Depot or str): an instance of Depot or a string containing the
            full path of where the root folder should be located for file
            output. Once a Depot instance is created, it is automatically made
            available to all other SimpleClass subclasses that are instanced in
            the future. If 'depot' is not passed, a default Depot instance will
            be created.
        ingredients (Ingredients, DataFrame, or str): an instance of
            Ingredients, a string containing the full file path of where a data
            file for a pandas DataFrame is located, or a string containing a
            file name in the default data folder, as defined in the Depot
            instance.
        infer_types (bool): whether values in 'options' are converted to
            other datatypes (True) or left as strings (False).

    """
    name: str = 'idea'
    options: object = None
    depot: object = None
    ingredients: object = None
    infer_types: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __add__(self, other):
        """Adds new settings using the 'update' method.

        Args:
            other (Idea, dict, or str): an Idea instance, a nested dictionary,
                or a file path to a configparser-compatible file.

        """
        self.update(new_settings = other)
        return self

    def __contains__(self, item):
        """Returns whether item is in 'options'.

        Args:
            item (str): key to be checked for a match in 'options'.

        """
        return item in self.options

    def __delitem__(self, key):
        """Removes a dictionary section if 'key' matches the name of a section.

        Otherwise, it will remove all entries with 'key' inside the various
        sections of the 'options' dictionary.

        Args:
            key (str): the name of the dictionary key or section to be deleted.

        Raises:
            KeyError: if 'key' not in 'options'.
        """
        try:
            del self.options[item]
        except KeyError:
            for section in list(self.options.keys()):
                try:
                    del section[key]
                except KeyError:
                    error = key + ' not found in Idea options dictionary'
                    raise KeyError(error)
        return self

    def __getitem__(self, key):
        """Returns a section of 'options' or key within a section.

        Args:
            key (str): the name of the dictionary key for which the value is
                sought.

        Returns:
            dict if 'key' matches a section in 'options'. If 'key'
                matches a key within a section, the value, which can be any of
                the supported datatypes is returned. If no match is found an
                empty dict is returned.

        """
        try:
            return self.options[item]
        except KeyError:
            for section in list(self.options.keys()):
                try:
                    return section[key]
                    break
                except KeyError:
                    continue
            return {}

    def __iadd__(self, other):
        """Adds new settings using the 'update' method.

        Args:
            other (Idea, dict, or str): an Idea instance, a nested dictionary,
                or a file path to a configparser-compatible file.

        """
        self.update(new_settings = other)
        return self

    def __iter__(self):
        """Returns iterable options dict items()."""
        return self.options.items()

    def __len__(self):
        """Returns length of 'options'."""
        return len(self.options)

    def __radd__(self, other):
        """Adds new settings using the 'update' method.

        Args:
            other (Idea, dict, or str): an Idea instance, a nested dictionary,
                or a file path to a configparser-compatible file.

        """
        self.update(new_settings = other)
        return self

    def __setitem__(self, section, dictionary):
        """Creates new key/value pair(s) in a specified section of
        'options'.

        Args:
            section (str): name of a section in 'options'.
            dictionary (dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'section' isn't a str or 'dictionary' isn't a dict.

        """
        if isinstance(section, str):
            if isinstance(dictionary, dict):
                if section in self.options:
                    self.options[section].update(dictionary)
                else:
                    self.options[section] = dictionary
            else:
                error = 'dictionary must be dict type'
                raise TypeError(error)
        else:
            error = 'section must be str type'
            raise TypeError(error)
        return self

    """ Private Methods """

    def _infer_types(self):
        """If 'infer_types' is True, values in 'options' are converted to
        the appropriate datatype.
        """
        if self.infer_types:
            for section, dictionary in self.options.items():
                for key, value in dictionary.items():
                    self.options[section][key] = self._typify(value)
        return self

    def _load_from_ini(self, file_path = None):
        """Creates a options dictionary from an .ini file."""
        if file_path:
            options_file = file_path
        else:
            options_file = self.options
        if os.path.isfile(options_file):
            options = ConfigParser(dict_type = dict)
            options.optionxform = lambda option: option
            options.read(options_file)
            self.options = dict(options._sections)
        else:
            error = 'options file ' + options_file + ' not found'
            raise FileNotFoundError(error)
        return self

    def _load_from_py(self, file_path = None):
        """Creates a options dictionary from an .py file.

        Todo:
            file_path(str): path to python module with 'options' dict
                defined.

        """
        if file_path:
            options_file = file_path
        else:
            options_file = self.options
        if os.path.isfile(options_file):
            self.options = getattr(import_module(options_file), 'options')
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
        """Stores the section or sections of the 'options' dictionary in
        the passed class instance as attributes to that class instance.

        Wildcard values of 'all', 'default', and 'none' are appropriately
        changed with the '_convert_wildcards' method.

        Args:
            instance(object): a class instance to which attributes should be
                added.
            sections(str or list(str)): the sections of 'options' which
                should be added to the instance.
            override(bool): if True, even existing attributes in instance will
                be replaced by 'options' key/value pairs. If False,
                current values in those similarly-named attributes will be
                maintained (unless they are None).

        Returns:
            instance with attribute(s) added.

        """
        for section in self.listify(sections):
            for key, value in self.options[section].items():
                setattr(instance, key, instance._convert_wildcards(value))
        return instance

    """ Core siMpLify Methods """

    def draft(self):
        """Creates 'options' dictionary and core attributes for Idea instance.

        Raises:
            AttributeError: if 'options' is None.
            TypeError: if 'options' is a path to a file that neither has an
                'ini' nor 'py' extension or if 'options' is neither a str nor a
                dict.

        """
        if self.options:
            if isinstance(self.options, str):
                if self.options.endswith('.ini'):
                    self.technique = self._load_from_ini
                elif self.options.endswith('.py'):
                    self.technique = self._load_from_py
                else:
                    error = 'options file must be .py or .ini file'
                    raise TypeError(error)
                if not os.path.isfile(os.path.abspath(self.options)):
                    self.options = os.path.join(os.getcwd(), self.options)
            elif not isinstance(self.options, dict):
                error = 'options must be dict or file path'
                raise TypeError(error)
        else:
            error = 'options dict or path needed to instance Idea'
            raise AttributeError(error)
        try:
            self.technique()
        except TypeError:
            pass
        self._infer_types()
        self._inject_base(attribute = 'idea')
        self.checks.extend(['depot', 'ingredients'])
        return self

    def publish(self):
        """Finalizes Idea and calls siMpLify controller."""
        self.inject(instance = self, sections = ['general'])
        super().publish()
        self.simplify = Simplify()
        self.simplify.publish(ingredients = self.ingredients)
        return self

    """ Python Dictionary Compatibility Methods """

    def update(self, new_settings):
        """Adds new settings to the options dictionary.

        Args:
           new_settings(dict, str, or Idea): can either be a dicti or Idea
               object containing new key/value pairs, or a str containing a
               file path from which new options options can be found.

        Raises:
            TypeError: if 'new_settings' is neither a dict, str, or Idea
                instance.
        """
        if isinstance(new_settings, dict):
            self.options.update(new_settings)
        elif isinstance(new_settings, str):
            if new_settings.endswith('.ini'):
                technique = self._load_from_ini
            elif new_settings.endswith('.py'):
                technique = self._load_from_py
            self.options.update(technique(file_path = new_settings))
        elif (hasattr(new_settings, 'options')
                and isinstance(new_settings.options, dict)):
            self.options.update(new_settings.options)
        else:
            error = 'new_options must be dict, Idea, or file path'
            raise TypeError(error)
        return self

