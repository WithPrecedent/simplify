"""
.. module:: idea
:synopsis: configuration made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from configparser import ConfigParser
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify
from simplify.core.utilities import typify


@dataclass
class Idea(MutableMapping):
    """Converts a data science idea into python.

    If 'configuration' is imported from a file, Idea creates a dictionary,
    converting dictionary values to appropriate datatypes, and stores portions
    of the 'configuration' dictionary as attributes in other classes. Idea is
    based on python's ConfigParser. It seeks to cure some of the shortcomings of
    the base ConfigParser including:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting items creates verbose code.
        3) It uses OrderedDict (python 3.6+ stepss regular dictionaries).

    To use the Idea class, the user can either pass to 'configuration':
        1) a file path, which will automatically be loaded into Idea;
        2) a file name which is located in the current working directory,
            which will automatically be loaded into Idea;
                                or,
        3) a prebuilt ConfigParser compatible nested dictionary.

    If 'infer_types' is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes (str, list, float,
    bool, and int are currently supported)

    Whichever option is chosen, the nested Idea dictionary is stored in the
    attribute 'configuration'. However, dictionary access methods can either be
    applied to the 'configuration' dictionary (e.g.,
    idea.configuration['general']) or an Idea instance (e.g., idea['general']).
    If using the dictionary 'update' method, it is better to apply it to the
    Idea instance because the Idea method is more flexible in handling different
    kinds of arguments.

    Users can add any key/value pairs from a section of the 'configuration'
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

    If the subclass wants the 'chef' settings as well, then the code should be:

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
                self.chef_steps = ['split', 'reduce', 'model']
                return self

    Regardless of the idea_sections added, all Idea settings can be similarly
    accessed using dict keys or local attributes. For example:

        self.workers.idea['general']['seed'] # typical dict access step

        self.workers.idea['seed'] # if no section or other key is named 'seed'

        self.seed # works because 'seed' is in the 'general' section

                            all return 43.

    Within the siMpLify ecosystem, settings of two types take on particular
    importance:
        'parameters': sections with the suffix '_parameters' are automatically
            linked to classes where the prefix matches the class's 'name' or
            'step'.
        'steps': settings with the suffix '_steps' are used to create
            iterable lists of actions to be taken (whether in parallel or
            serial).

    Because Idea uses ConfigParser, it only allows 2-level dictionaries. The
    desire for accessibility and simplicity dictated this limitation.

    Args:
        project (Optional['Project']): related Project or subclass instance.
            Defaults to None.
        configuration (Dict[str, Dict[str, Any]]): a two-level nested
            dictionary storing settings. Defaults to an empty dictionary.
        infer_types (Optional[bool]): whether values in 'configuration' are
            converted to other datatypes (True) or left alone (False). If
            'configuration' was imported, a False value will leave all values as
            strings. Defaults to True.

    """
    project: 'Project' = None
    configuration: Dict[str, Dict[str, Any]] = field(default_factory = dict)
    infer_types: Optional[bool] = True

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Validates 'configuration'.
        if not isinstance(self.configuration, (MutableMapping, dict)):
            self = create_idea(idea = self.configuration)
        elif self.infer_types:
            self._infer_types()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Union[Dict[str, Any], Any]:
        """Returns a section of the active dictionary or key within a section.

        Args:
            key (str): the name of the dictionary key for which the value is
                sought.

        Returns:
            Union[Dict[str, Any], Any]: dict if 'key' matches a section in
                the active dictionary. If 'key' matches a key within a section,
                the value, which can be any of the supported datatypes is
                returned.

        """
        try:
            return self.configuration[key]
        except KeyError:
            for section in list(self.configuration.keys()):
                try:
                    return self.configuration[section][key]
                    break
                except KeyError:
                    pass

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        """Creates new key/value pair(s) in a section of the active dictionary.

        Args:
            key (str): name of a section in the active dictionary.
            value (Dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'key' isn't a str or 'value' isn't a dict.

        """
        try:
            self.configuration[key].update(value)
        except KeyError:
            try:
                self.configuration[key] = value
            except TypeError:
                raise TypeError(
                    'item must be a str and value must be a dict type')
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'configuration'.

        Args:
            key (str): name of key in 'configuration'.

        """
        try:
            del self.configuration[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'configuration'.

        Returns:
            Iterable stored in 'configuration'.

        """
        return iter(self.configuration.items())

    def __len__(self) -> int:
        """Returns length of 'configuration'.

        Returns:
            Integer: length of 'configuration'.

        """
        return len(self.configuration)

    """ Private Methods """

    def _add_special(self, instance: object, attribute: str) -> object:
        """Injects 'workers' into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.
            attribute (str): the name of attribute to check or add to 'instance'
                which also match the suffix in 'configuration'.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """
        if not hasattr(instance, attribute) or not getattr(instance, attribute):
            name = instance.name
            setattr(instance, attribute, listify(
                self.configuration[name]['_'.join([name, attribute])],
                use_empty = True))
        return instance

    def _add_parameters(self, instance: object) -> object:
        """Injects 'parameters' into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """

        try:
            instance.parameters == self.configuration['_'.join(
                [instance.name, 'parameters'])]
        except (AttributeError, KeyError):
            try:
                instance.parameters == self.configuration['_'.join(
                    [instance.technique, 'parameters'])]
            except (AttributeError, KeyError):
                pass
        return instance

    def _infer_types(self):
        """Converts stored values to appropriate datatypes.

        The method supports one level of nesting, but doesn't use recursion to
        avoid supporting more.

        """
        new_bundle = {}
        for key, value in self.configuration.items():
            if isinstance(value, dict):
                inner_bundle = {}
                for inner_key, inner_value in value.items():
                    inner_bundle[inner_key] = typify(inner_value)
                new_bundle[key] = inner_bundle
            else:
                new_bundle[key] = typify(value)
        self.configuration = new_bundle
        return self

    """ Core siMpLify Methods """

    def apply(self, instance: object) -> object:
        """Injects appropriate attributes into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """
        sections = ['general']
        try:
            sections.append(instance.name)
        except AttributeError:
            pass
        try:
            sections.extend(listify(instance.idea_sections))
        except AttributeError:
            pass
        if hasattr(instance, 'parameters') and not instance.parameters:
            instance = self._add_parameters(instance = instance)
        for section in sections:
            try:
                for key, value in self.configuration[section].items():
                    if key.endswith('workers'):
                        instance = self._add_special(
                            instance = instance,
                            attribute = 'workers')
                    elif key.endswith('steps'):
                        instance = self._add_special(
                            instance = instance,
                            attribute = 'steps')
                    elif (not hasattr(instance, key)
                            or not getattr(instance, key)):
                        setattr(instance, key, value)
            except KeyError:
                pass
        return instance


""" Validator Function """

def create_idea(idea: Union[Dict[str, Dict[str, Any]], 'Idea']) -> 'Idea':
    """Creates an Idea instance from passed argument.

    Args:
        idea (Union[Dict[str, Dict[str, Any]], 'Idea']): a dict, a str file path
            to an ini, csv, or py file with settings, or an Idea instance with a
            'configuration' attribute.

    Returns:
        Idea instance, properly configured.

    Raises:
        TypeError: if 'idea' is neither a dict, str, nor Idea instance.

    """
    def _load_from_csv(file_path: str) -> Dict[str, Any]:
        """Creates a configuration dictionary from a .csv file.

        Args:
            file_path (str): path to siMpLify-compatible .csv file.

        Returns:
            Dict[str, Any] of settings.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            configuration = pd.read_csv(file_path, dtype = 'str')
            return configuration.to_dict(orient = 'list')
        except FileNotFoundError:
            raise FileNotFoundError(' '.join(['configuration file',
                file_path, 'not found']))


    def _load_from_ini(file_path: str) -> Dict[str, Any]:
        """Creates a configuration dictionary from an .ini file.

        Args:
            file_path (str): path to configparser-compatible .ini file.

        Returns:
            Dict[str, Any] of configuration.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            configuration = ConfigParser(dict_type = dict)
            configuration.optionxform = lambda option: option
            configuration.read(file_path)
            return dict(configuration._sections)
        except FileNotFoundError:
            raise FileNotFoundError(' '.join(['configuration file',
                file_path, 'not found']))

    def _load_from_py(file_path: str) -> Dict[str, Any]:
        """Creates a configuration dictionary from a .py file.

        Args:
            file_path (str): path to python module with '__dict__' dict defined.

        Returns:
            Dict[str, Any] of configuration.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            return getattr(import_module(file_path), '__dict__')
        except FileNotFoundError:
            raise FileNotFoundError(' '.join(['configuration file',
                file_path, 'not found']))

    if isinstance(idea, Idea):
        return idea
    elif isinstance(idea, (dict, MutableMapping)):
        return Idea(configuration = idea)
    elif isinstance(idea, (str, Path)):
        extension = str(Path(idea).suffix)[1:]
        configuration = locals()['_'.join(['_load_from', extension])](
            file_path = idea)
        return Idea(configuration = configuration)
    else:
        raise TypeError('idea must be Idea, str, or nested dict type')
