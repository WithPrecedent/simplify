"""
.. module:: idea
:synopsis: configuration made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from collections import ChainMap
from collections.abc import MutableMapping
from configparser import ConfigParser
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from pathlib import Path
import re
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import pandas as pd

from simplify.core.base import SimpleContainer
import simplify.core.defaults as simplify_defaults
from simplify.core.repository import Repository
from simplify.core.utilities import listify
from simplify.core.utilities import typify


@dataclass
class Idea(SimpleContainer, MutableMapping):
    """Converts a data science idea into a python object.

    'Idea' uses a modified version of the Borg design pattern outlined by Alex
    Martelli, described here: http://www.aleax.it/5ep.html.

    This allows every imported instance of 'Idea' to share the same class
    variable 'configuration' dictionary containing settings.

    If 'configuration' is imported from a file, 'Idea' creates a dictionary,
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

        [analyst]
        analyst_steps = split, reduce, model

    'verbose' and 'file_type' will automatically be added to every siMpLify
    class because they are located in the 'general' section. If a subclass
    wants attributes from the files section, then the following line should
    appear in __post_init__ before calling super().__post_init__:

        self.idea_sections = ['files']

    If the subclass wants the 'analyst' settings as well, then the code should
    be:

        self.idea_sections = ['files', 'analyst']

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
                self.analyst_steps = ['split', 'reduce', 'model']
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
        configuration (Dict[str, Dict[str, Any]]): a two-level nested
            dictionary storing settings. Defaults to an empty dictionary.
        infer_types (Optional[bool]): whether values in 'configuration' are
            converted to other datatypes (True) or left alone (False). If
            'configuration' was imported, a False value will leave all values as
            strings. Defaults to True.
        _shared_state (ClassVar[Dict[str, Any]]): shared state of all class
            instances. Defaults to an empty dictionary.

    """
    configuration: Optional[Dict[str, Dict[str, Any]]] = field(
        default_factory = dict)
    infer_types: Optional[bool] = True
    # _shared_state: ClassVar[Dict[str, Any]] = field(default = {})

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Validates 'configuration'.
        if isinstance(self.configuration, (str, Path)):
            self.configuration = self.create(idea = self.configuration)
        # Infers types for values in 'configuration', if option selected.
        if self.infer_types:
            self._infer_types()
        # Adds 'simplify_defaults' as backup settings to 'configuration'.
        self._chain_defaults()
        # Implements Borg pattern so that all 'Idea' instances will share the
        # same state.
        # self.__dict__ = self._shared_state
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            idea: Union[
                str, Path, Dict[str, Dict[str, Any]], 'Idea']) -> 'Idea':
        """Creates an 'Idea' instance from passed argument.

        Args:
            idea (Union[Dict[str, Path, Dict[str, Any]], 'Idea']): a dict, a str
                file path to an ini, csv, or py file with settings, or an Idea
                instance with a 'configuration' attribute.

        Returns:
            'Idea' instance, properly configured.

        Raises:
            TypeError: if 'idea' is neither a dict, str, nor 'Idea' instance.

        """

        def _load_from_csv(file_path: str) -> Dict[str, Any]:
            """Creates a configuration dictionary from a .csv file.

            Args:
                file_path (str): path to siMpLify-compatible .csv file.

            Returns:
                Dict[str, Any] of settings.

            Raises:
                FileNotFoundError: if the file_path does not correspond to a
                    file.

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
                FileNotFoundError: if the file_path does not correspond to a
                    file.

            """
            try:
                configuration = ConfigParser(dict_type = dict)
                configuration.optionxform = lambda option: option
                configuration.read(str(file_path))
                return dict(configuration._sections)
            except FileNotFoundError:
                raise FileNotFoundError(' '.join(['configuration file',
                    file_path, 'not found']))

        def _load_from_py(file_path: str) -> Dict[str, Any]:
            """Creates a configuration dictionary from a .py file.

            Args:
                file_path (str): path to python module with '__dict__' dict
                    defined.

            Returns:
                Dict[str, Any] of configuration.

            Raises:
                FileNotFoundError: if the file_path does not correspond to a
                    file.

            """
            try:
                return getattr(import_module(file_path), '__dict__')
            except FileNotFoundError:
                raise FileNotFoundError(' '.join(['configuration file',
                    file_path, 'not found']))

        if isinstance(idea, Idea):
            return idea
        elif isinstance(idea, (dict, MutableMapping)):
            return cls(configuration = idea)
        elif isinstance(idea, (str, Path)):
            extension = str(Path(idea).suffix)[1:]
            load_method = locals()['_'.join(['_load_from', extension])]
            return cls(configuration = load_method(file_path = idea))
        elif idea is None:
            return cls()
        else:
            raise TypeError('idea must be Idea, str, or nested dict type')

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
        return iter(self.configuration)

    def __len__(self) -> int:
        """Returns length of 'configuration'.

        Returns:
            Integer: length of 'configuration'.

        """
        return len(self.configuration)

    """ Private Methods """

    def _chain_defaults(self) -> None:
        """Creates set of mappings for siMpLify settings lookup."""
        defaults = {}
        for key, attribute in simplify_defaults.__dict__.items():
            defaults[key.lower()] = attribute
        self.configuration = ChainMap(self.configuration, defaults)
        return self

    def _get_parameters(self, instance: object) -> Dict[str, Any]:
        """Returns 'parameters' dictionary appropriate to 'instance'.

        Args:
            instance (object): siMpLify class with 'name' attribute to find
                matching items in 'configuration'.

        Returns:
            Dict[str, Any]: parameters dictionary stored in 'configuration'.

        """
        try:
            return self.configuration['_'.join([instance.name, 'parameters'])]
        except (AttributeError, KeyError):
            try:
                return self.configuration['_'.join(
                    [instance.technique, 'parameters'])]
            except (AttributeError, KeyError):
                return {}

    def _get_special(self, section: str, prefix: str, suffix: str) -> List[str]:
        """Returns list of strings from appropriate item in 'configuration'.

        Args:
            section (str): outer key for item in 'configuration'.
            prefix (str): prefix to inner key for item in 'configuration'.
            suffix (str): suffix of inner key for item in 'configuration'.

        Returns:
            List[str]: item from 'configuration.

        """
        try:
            return listify(self.configuration[section][f'{prefix}_{suffix}'])
        except (KeyError, AttributeError):
            return None

    # def _get_special(self, instance: object, suffix: str) -> List[str]:
    #     """Returns list of strings from appropriate item in 'configuration'.

    #     Args:
    #         instance (object): siMpLify class with 'name' attribute to find
    #             matching item in 'configuration'.
    #         suffix (str): suffix of item in 'configuration'.

    #     Returns:
    #         List[str]: item from 'configuration.

    #     """
    #     try:
    #         return listify(self.configuration[instance.name]['_'.join(
    #             [instance.name, suffix])])
    #     except (KeyError, AttributeError):
    #         return None

    def _get_techniques(self, instance: object) -> Dict[str, Dict[str, None]]:
        """Returns nested dictionary of techniques.

        Args:
            instance (object): siMpLify class with 'name' attribute to find
                matching items in 'configuration'.

        Returns:
            Dict[str, Dict[str, None]]: techniques dictionary prepared to be
                loaded into a 'Repository' instance.

        """
        contents = {}
        try:
            for key, value in self.configuration[instance.name].items():
                if (key.endswith('_techniques')
                        and not value in [None, 'none', 'None']):
                    step = key.replace('_techniques', '')
                    contents[step] = {}
                    for technique in listify(value):
                        contents[step][technique] = None
            return Repository(contents = contents)
        except (KeyError, AttributeError):
            return None

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

    def _inject(self,
            instance: object,
            attribute: str,
            value: Any,
            overwrite: bool) -> object:
        """Adds attribute to 'instance' based on conditions.

        Args:
            instance (object): siMpLify class instance to be modified.
            attribute (str): name of attribute to inject.
            value (Any): value to assign to attribute.
            overwrite (Optional[bool]): whether to overwrite a local attribute
                in 'instance' if there are values stored in that attribute.
                Defaults to False.

        Returns:
            object: with attribute possibly injected.

        """
        if (not hasattr(instance, attribute)
                or not getattr(instance, attribute)
                or overwrite):
            setattr(instance, attribute, value)
        return instance

    """ Public Methods """

    def inject_attributes(self, instance: object,
            overwrite: Optional[bool] = False) -> object:
        """Injects appropriate items into 'instance' from 'configuration'.

        Args:
            instance (object): siMpLify class instance to be modified.
            overwrite (Optional[bool]): whether to overwrite a local attribute
                in 'instance' if there are values stored in that attribute.
                Defaults to False.

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
        for section in sections:
            try:
                for key, value in self.configuration[section].items():
                    self._inject(
                        instance = instance,
                        attribute = key,
                        value = value,
                        overwrite = overwrite)
            except KeyError:
                pass
        return instance

    def inject_parameters(self,
            instance: object,
            overwrite: Optional[bool] = False) -> object:
        """Injects 'parameters' into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.
            overwrite (Optional[bool]): whether to overwrite a local attribute
                in 'instance' if there are values stored in that attribute.
                Defaults to False.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """
        self._inject(
            instance = instance,
            attribute = 'parameters',
            value = self._get_parameters(instance = instance),
            overwrite = overwrite)
        return instance

    def inject_steps(self, instance: object,
            overwrite: Optional[bool] = False) -> object:
        """Injects 'steps' into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.
            overwrite (Optional[bool]): whether to overwrite a local attribute
                in 'instance' if there are values stored in that attribute.
                Defaults to False.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """
        self._inject(
            instance = instance,
            attribute = 'steps',
            value = self._get_special(instance = instance, suffix = 'steps'),
            overwrite = overwrite)
        return instance

    def inject_techniques(self,
            instance: object,
            overwrite: Optional[bool] = False) -> object:
        """Injects 'techniques' into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.
            overwrite (Optional[bool]): whether to overwrite a local attribute
                in 'instance' if there are values stored in that attribute.
                Defaults to False.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """
        self._inject(
            instance = instance,
            attribute = 'techniques',
            value = self._get_techniques(instance = instance),
            overwrite = overwrite)
        return instance

    def inject_workers(self,
            instance: object,
            overwrite: Optional[bool] = False) -> object:
        """Injects 'techniques' into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.
            overwrite (Optional[bool]): whether to overwrite a local attribute
                in 'instance' if there are values stored in that attribute.
                Defaults to False.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """
        self._inject(
            instance = instance,
            attribute = 'workers',
            value = self._get_special(instance = instance, suffix = 'workers'),
            overwrite = overwrite)
        return instance

    """ Core siMpLify Methods """

    def add(self, section: str, dictionary: Dict[str, Any]) -> None:
        """Adds entry to 'configuration'.

        Args:
            section (str): name of section to add 'dictionary' to.
            dictionary (Dict[str, Any]): dict to add to 'section'.

        """
        if section in self.configuration:
            self.configuration[section].update(dictionary)
        else:
            self.configuration[section] = dictionary
        return self

    def apply(self,
            instance: object,
            inject_specials: Optional[bool] = True,
            overwrite: Optional[bool] = False) -> object:
        """Injects appropriate attributes into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.
            inject_specials (Optional[bool]): whether to add 'parameters',
                'steps', 'techniques', or 'workers' to 'instance'. Defaults to
                True.
            overwrite (Optional[bool]): whether to overwrite a local attribute
                in 'instance' if there are values stored in that attribute.
                Defaults to False.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """
        # Adds special attributes appropraite to instance.
        if inject_specials:
            for special in ['parameters', 'steps', 'techniques', 'workers']:
                if hasattr(instance, special):
                    getattr(self, '_'.join(['inject', special]))(
                        instance = instance,
                        overwrite = overwrite)
        # Adds 'general' and other appropriate items to 'instance' as attributes
        # from 'configuration.
        instance = self.inject_attributes(
            instance = instance,
            overwrite = overwrite)
        return instance

    """ Special Access Methods """

    def get_packages(self, section: str) -> List[str]:
        return self._get_special(
            section = section,
            prefix = section,
            suffix = 'packages')

    def get_steps(self, section: str) -> List[str]:
        return self._get_special(
            section = section,
            prefix = section,
            suffix = 'step')

    def get_techniques(self, section: str) -> List[str]:
        techniques = {}
        for key, value in self.configuration[section].items():
            if '_techniques' in key:
                new_key = key.partition('_techniques')[0]
                techniques[new_key] = self._get_special(
                    section = section,
                    prefix = new_key,
                    suffix = 'techniques')
        return techniques

    def get_parameters(self, step: str, technique: str) -> Dict[str, Any]:
        try:
            return self.configuration[f'{technique}_parameters']
        except KeyError:
            try:
                return self.configuration[f'{step}_parameters']
            except KeyError:
                raise KeyError(
                    f'parameters for {step} and {technique} not found')