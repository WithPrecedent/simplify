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
from typing import Any, Dict, Iterable, List, Union

from simplify.core.base import SimpleClass
from simplify.core.utilities import listify


@dataclass
class Idea(SimpleClass):
    """Converts a data science idea into python.

    If 'options' are imported from a file, Idea creates a nested dictionary,
    converting dictionary values to appropriate datatypes, and stores portions
    of the 'options' dictionary as attributes in other classes. Idea is based
    on python's ConfigParser. It seeks to cure some of the shortcomings of the
    base ConfigParser including:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting items creates verbose code.
        3) It uses OrderedDict (python 3.6+ orders regular dictionaries).

    To use the Idea class, the user can either pass to 'options':
        1) a file path, which will automatically be loaded into Idea;
        2) a file name which is located in the current working directory,
            which will automatically be loaded into Idea;
                                or,
        3) a prebuilt ConfigParser compatible nested dictionary.

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

    If the subclass wants the chef settings as well, then the code should be:

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

        self.seed # works because 'seed' is in the 'general' section

                            all return 43.

    Within the siMpLify ecosystem, settings of two types take on particular
    importance and are automatically injected to certain classes:
        'parameters': sections with the suffix '_parameters' are automatically
            added to classes where the prefix matches the class's 'name' or
            'technique'.
        'techniques': settings with the suffix '_techniques' are automatically
            added to classes with the prefix as the name of the attribute.
            These techniques are stored in lists used to create the
            permutations and/or sequences of possible techniques in subclasses
            of SimplePlan.

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
        infer_types (bool): whether values in 'options' are converted to
            other datatypes (True) or left as strings (False).

    """
    name: str = 'idea'
    options: object = None
    infer_types: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __add__(self, other: Union[Dict, str, 'Idea']) -> None:
        """Adds new settings using the 'update' method.

        Args:
            other (Idea, dict, or str): an Idea instance, a nested dictionary,
                or a file path to a configparser-compatible file.

        """
        self.update(new_settings = other)
        return self

    def __contains__(self, item: str) -> bool:
        """Returns whether item is in 'options'.

        Args:
            item (str): key to be checked for a match in 'options'.

        """
        return item in self.options

    def __delitem__(self, key: str) -> None:
        """Removes a dictionary section if 'key' matches the name of a section.

        Otherwise, it will remove all entries with 'key' inside the various
        sections of the 'options' dictionary.

        Args:
            key (str): the name of the dictionary key or section to be deleted.

        Raises:
            KeyError: if 'key' not in 'options'.
        """
        try:
            del self.options[key]
        except KeyError:
            for section in list(self.options.keys()):
                try:
                    del section[key]
                except KeyError:
                    error = key + ' not found in Idea options dictionary'
                    raise KeyError(error)
        return self

    def __getitem__(self, key: str) -> Any:
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
            return self.options[key]
        except KeyError:
            for section in list(self.options.keys()):
                try:
                    return section[key]
                    break
                except KeyError:
                    continue
            return {}

    def __iadd__(self, other: Union[Dict, str, 'Idea']) -> None:
        """Adds new settings using the 'update' method.

        Args:
            other (Idea, dict, or str): an Idea instance, a nested dictionary,
                or a file path to a configparser-compatible file.

        """
        self.update(new_settings = other)
        return self

    def __radd__(self, other: Union[Dict, str, 'Idea']) -> None:
        """Adds new settings using the 'update' method.

        Args:
            other (Idea, dict, or str): an Idea instance, a nested dictionary,
                or a file path to a configparser-compatible file.

        """
        self.update(new_settings = other)
        return self

    def __setitem__(self, section: str, dictionary: Dict) -> None:
        """Creates new key/value pair(s) in a specified section of
        'options'.

        Args:
            section (str): name of a section in 'options'.
            dictionary (Dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'section' isn't a str or 'dictionary' isn't a dict.

        """
        try:
            self.options[section].update(dictionary)
        except KeyError:
            try:
                self.options[section] = dictionary
            except TypeError:
                try:
                    self.options[section] = dictionary
                except TypeError:
                    error = ''.join('section must be str and dictionary must',
                                    ' be dict type')
                    raise TypeError(error)
        return self

    """ Private Methods """

    def _add_settings(self, settings: Union[Dict, str, 'Idea'] = None) -> None:
        """Adds new settings to the options dictionary.

        Args:
           settings (dict, str, or Idea): can either be a dict or Idea
               object containing new key/value pairs, or a str containing a
               file path from which new options options can be found.

        Raises:
            TypeError: if 'new_settings' is neither a dict, str, or Idea
                instance.

        """
        if settings is None:
            settings = self.options
        try:
            self.options.update(settings)
        except AttributeError:
            if settings.endswith('.ini'):
                technique = self._load_from_ini
            elif settings.endswith('.py'):
                technique = self._load_from_py
            technique(file_path = settings)
        except TypeError:
            try:
                self.options.update(settings.options)
            except TypeError:
                error = 'options settings must be dict, Idea, or file path'
                raise TypeError(error)
        return self

    def _infer_types(self, settings: Dict = None) -> None:
        """If 'infer_types' is True, values in 'options' are converted to
        the appropriate datatype.
        """
        if self.infer_types:
            if settings is None:
                settings = self.options
            for section, dictionary in settings.items():
                for key, value in dictionary.items():
                    self.options[section][key] = self._typify(value)
        return self

    def _load_from_ini(self, file_path: str = None) -> None:
        """Creates a options dictionary from an .ini file.

        Args:
            file_path (str): path to configparser-compatible .ini file.

        """
        try:
            options = ConfigParser(dict_type = dict)
            options.optionxform = lambda option: option
            options.read(file_path)
            self.options = dict(options._sections)
        except FileNotFoundError:
            error = 'options file ' + file_path + ' not found'
            raise FileNotFoundError(error)
        return self

    def _load_from_py(self, file_path: str = None) -> None:
        """Creates a options dictionary from an .py file.

        Args:
            file_path (str): path to python module with 'options' dict defined.

        """
        try:
            self.options = getattr(import_module(file_path), 'options')
        except FileNotFoundError:
            error = 'options file ' + file_path + ' not found'
            raise FileNotFoundError(error)
        return self

    @staticmethod
    def _numify(variable: str) -> Union[int, float, str]:
        """Attempts to convert 'variable' to a numeric type.

        Args:
            variable (str): variable to be converted.

        Returns
            variable (int, float, str) converted to numeric type, if possible.

        """
        try:
            return int(variable)
        except ValueError:
            try:
                return float(variable)
            except ValueError:
                return variable

    def _typify(self, variable: str) -> Union[List, int, float, bool, str]:
        """Converts stingsr to appropriate, supported datatypes.

        The method converts strings to list (if ', ' is present), int, float,
        or bool datatypes based upon the content of the string. If no
        alternative datatype is found, the variable is returned in its original
        form.

        Args:
            variable (str): string to be converted to appropriate datatype.

        Returns:
            variable (str, list, int, float, or bool): converted variable.
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

    def inject(self, instance: SimpleClass, sections: Union[List[str], str], 
               override: bool = False) -> SimpleClass:
        """Stores the section or sections of the 'options' dictionary in
        the passed class instance as attributes to that class instance.

        Wildcard values of 'all', 'default', and 'none' are appropriately
        changed with the '_convert_wildcards' method.

        Args:
            instance (object): a class instance to which attributes should be
                added.
            sections (str or list(str)): the sections of 'options' which
                should be added to the instance.
            override (bool): if True, even existing attributes in instance will
                be replaced by 'options' key/value pairs. If False,
                current values in those similarly-named attributes will be
                maintained (unless they are None).

        Returns:
            instance with attribute(s) added.

        """
        for section in listify(sections):
            try:
                for key, value in self.options[section].items():
                    # if not instance.exists(key) or override:
                    setattr(instance, key, value)
            except KeyError:
                pass
        return instance

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates 'options' dictionary and core attributes for Idea instance.

        Raises:
            AttributeError: if 'options' is None.
            TypeError: if 'options' is a path to a file that neither has an
                'ini' nor 'py' extension or if 'options' is neither a str nor a
                dict.

        """
        self._add_settings()
        self._infer_types()
        super().draft()
        self.publish()
        return self

    def publish(self) -> None:
        """Finalizes Idea and calls siMpLify controller."""
        self = self.inject(instance = self, sections = ['general'])
        return self

    """ Python Dictionary Compatibility Methods """

    def update(self, new_settings: Union[Dict, str, 'Idea']) -> None:
        """Adds new settings to the options dictionary.

        Args:
           new_settings (dict, str, or Idea): can either be a dict or Idea
               object containing new key/value pairs, or a str containing a
               file path from which new options options can be found.

        Raises:
            TypeError: if 'new_settings' is neither a dict, str, or Idea
                instance.

        """
        self._add_settings(settings = new_settings)
        self._infer_types()
        return self