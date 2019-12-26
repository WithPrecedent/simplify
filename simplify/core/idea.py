"""
.. module:: idea
:synopsis: configuration made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

from simplify.core.base import SimpleSettings
from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify
from simplify.core.utilities import typify


@dataclass
class Idea(SimpleSettings):
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

        self.options.idea['general']['seed'] # typical dict access step

        self.options.idea['seed'] # if no section or other key is named 'seed'

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
        configuration (Union[Dict[str, Dict[str, Any]], str]): a file path or
            two-level nested dictionary storing settings. If a file path is
            passed, a nested dictionary will automatically be created from the
            file and stored in 'configuration'. Defaults to an empty dictionary.
        infer_types (Optional[bool]): whether values in 'configuration' are
            converted to other datatypes (True) or left alone (False). If
            'configuration' was imported, a False value will leave all values as
            strings. Defaults to True.

    """
    project: 'Project'
    configuration: Union[Dict[str, Dict[str, Any]], str] = field(
        default_factory = dict)
    infer_types: Optional[bool] = True

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        self.draft()
        return self

    """ Dunder Methods """

    def __getitem__(self, item: str) -> Any:
        """Returns a section of the active dictionary or key within a section.

        Args:
            item (str): the name of the dictionary key for which the value is
                sought.

        Returns:
            Union[Dict[str, Any], Any]: dict if 'item' matches a section in
                the active dictionary. If 'item' matches a key within a section,
                the value, which can be any of the supported datatypes is
                returned.

        Raises:
            KeyError: if 'item' not in the active dictionary.

        """
        try:
            return self.configuration[item]
        except KeyError:
            for section in list(self.configuration.keys()):
                try:
                    return section[item]
                    break
                except KeyError:
                    pass
            raise KeyError(' '.join([item, 'not found in Idea']))

    def __setitem__(self, item: str, value: Dict[str, Any]) -> None:
        """Creates new key/value pair(s) in a section of the active dictionary.

        Args:
            item (str): name of a section in the active dictionary.
            value (Dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'item' isn't a str or 'value' isn't a dict.

        """
        try:
            self.configuration[item].update(value)
        except KeyError:
            try:
                self.configuration[item] = value
            except TypeError:
                raise TypeError(' '.join(['item must be a str and value must',
                    'be a dict type']))
        return self

    """ Private Methods """

    def _are_parameters(self, instance: object, section: str) -> bool:
        """Returns whether value stores matching parameters for instance.

        Args:
            instance (object): a class instance to which attributes should be
                added.
            section (str): name of a section of the configuration settings.

        Returns:
            bool: whether the section includes parameters and if those
                parameters correspond to the class name or step name.

        """
        if '_parameters' in section:
            try:
                return (instance.name == '_'.join(
                    [instance.name, '_parameters']))
            except AttributeError:
                try:
                    return (instance.technique == '_'.join(
                        [instance.technique , '_parameters']))
                except AttributeError:
                    pass
            return False
        else:
            return False

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

    def draft(self) -> None:
        """Creates 'configuration' dictionary from 'passed_configuration'."""
        if self.infer_types:
            self._infer_types()
        return self

    def apply(self, instance: object) -> object:
        """Injects appropriate attributes into 'instance'.

        Args:
            instance (object): siMpLify class instance to be modified.

        Returns:
            instance (object): siMpLify class instance with modifications made.

        """
        sections = ['general', instance.name]
        try:
            sections.extend(listify(instance.idea_sections))
        except AttributeError:
            pass
        for section in sections:
            try:
                for key, value in self.project.idea[section].items():
                    if not hasattr(instance, key):
                        setattr(instance, key, value)
            except KeyError:
                pass
        return instance