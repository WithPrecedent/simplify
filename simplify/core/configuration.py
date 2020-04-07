"""
.. module:: configuration
:synopsis: configuration made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import configparser
import dataclasses
import importlib
import pathlib
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core import defaults
from simplify.core import utilities


@dataclasses.dataclass
class Idea(collections.abc.MutableMapping):
    """Converts an idea into a python nested dictionary.

    To use the Idea class, the user can either pass to the 'create' classmethod:
        1) a file path, which will automatically be loaded into Idea;
        2) a file name which is located in the current working directory,
            which will automatically be loaded into Idea;
                                or,
        3) a prebuilt compatible dictionary.

    If 'contents' is imported from a file, 'Idea' creates a dictionary and can
    convert the dictionary values to appropriate datatypes. Currently supported
    file types are: .ini and .py.

    For .ini files, Idea uses python's ConfigParser. It seeks to cure some of
    the shortcomings of the base ConfigParser including:
        1) All values in ConfigParser are strings by default.
        2) The nested structure for getting and settingitems creates verbose
            code.
        3) It uses OrderedDict (python 3.6+ orders regular dictionaries).

    With .py files, users are free to set any datatypes in the original python
    file.

    If 'infer_types' is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes (str, list, float,
    bool, and int are currently supported)

    Whichever option is chosen, the nested Idea dictionary is stored in the
    attribute 'contents'. However, dictionary access methods can either be
    applied to the 'contents' dictionary (e.g., idea.contents['general']) or an
    Idea instance (e.g., idea['general']).

    Within the siMpLify ecosystem, settings of four types take on particular
    importance:
        'parameters': sections with the suffix '_parameters' are automatically
            linked to classes where the prefix matches the class's 'name' or
            'step'.
        'workers':
        'steps': settings with the suffix '_steps' are used to create
            iterable lists of actions to be taken (whether in parallel or
            serial).
        'techniques':

    Because Idea uses ConfigParser, it only allows 1- or 2-level dictionaries.
    The desire for accessibility and simplicity dictated this limitation.

    Args:
        contents (Union[str, pathlib.Path, Dict[str, Any], 'Idea']): a
            dict, a str file path to an .ini or .py file with settings, or
            an Idea instance with a 'contents' attribute. Defaults to an
            empty dictionary.
        infer_types (Optional[bool]): whether values in 'contents' are converted
            to other datatypes (True) or left alone (False). If 'contents' was
            imported from an .ini file, a False value will leave all values as
            strings. Defaults to True.

    """
    contents: Optional[Union[
        str, pathlib.Path, Dict[str, Any], 'Idea']] = dataclasses.field(
            default_factory = dict)
    infer_types: Optional[bool] = True

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Configures 'contents' to proper type.
        if not isinstance(contents, dict):
            self = IdeaLoader(idea = self.contents)
        # Infers types for values in 'contents', if option selected.
        if self.infer_types:
            self._infer_types()
        # Adds default settings as backup settings to 'contents'.
        self._chain_defaults()
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
            return self.contents[key]
        except KeyError:
            for section in list(self.contents.keys()):
                try:
                    return self.contents[section][key]
                except KeyError:
                    pass
            raise KeyError(f'{key} is not found in {self.name}')

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        """Creates new key/value pair(s) in a section of the active dictionary.

        Args:
            key (str): name of a section in the active dictionary.
            value (Dict): the dictionary to be placed in that section.

        Raises:
            TypeError if 'key' isn't a str or 'value' isn't a dict.

        """
        try:
            self.contents[key].update(value)
        except KeyError:
            try:
                self.contents[key] = value
            except TypeError:
                raise TypeError(
                    'key must be a str and value must be a dict type')
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'contents'.

        Args:
            key (str): name of key in 'contents'.

        """
        try:
            del self.contents[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'contents'.

        Returns:
            Iterable stored in 'contents'.

        """
        return iter(self.contents)

    def __len__(self) -> int:
        """Returns length of 'contents'.

        Returns:
            Integer: length of 'contents'.

        """
        return len(self.contents)

    """ Private Methods """

    def _chain_defaults(self) -> None:
        """Creates set of mappings for siMpLify settings lookup."""
        defaults = {
            k.lower(): v
            for k, v in defaults.simplify_defaults.__dict__.items()}
        self.contents = collections.ChainMap(self.contents, defaults)
        return self

    def _infer_types(self):
        """Converts stored values to appropriate datatypes.

        The method supports one level of nesting, but doesn't use recursion to
        avoid supporting more.

        """
        new_bundle = {}
        for key, value in self.contents.items():
            if isinstance(value, dict):
                inner_bundle = {
                    inner_key: utilities.typify(inner_value)
                    for inner_key, inner_value in value.items()}
                new_bundle[key] = inner_bundle
            else:
                new_bundle[key] = utilities.typify(value)
        self.contents = new_bundle
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

    """ Core siMpLify Methods """

    def add(self, section: str, dictionary: Dict[str, Any]) -> None:
        """Adds entry to 'contents'.

        Args:
            section (str): name of section to add 'dictionary' to.
            dictionary (Dict[str, Any]): dict to add to 'section'.

        """
        if section in self:
            self[section].update(dictionary)
        else:
            self[section] = dictionary
        return self

    """ Special Access Methods """

    def get_workers(self, section: str) -> List[str]:
        """Returns 'workers' list appropriate to 'section'

        Args:
            section (str): name of section in 'contents' to search for
                steps.

        Returns:
            List[str]: steps list stored in 'contents'.

        """
        return self._get_special(
            section = section,
            prefix = section,
            suffix = 'workers')

    def get_steps(self, section: str) -> List[str]:
        """Returns 'steps' list appropriate to 'section'

        Args:
            section (str): name of section in 'contents' to search for
                steps.

        Returns:
            List[str]: steps list stored in 'contents'.

        """
        return self._get_special(
            section = section,
            prefix = section,
            suffix = 'steps')

    def get_techniques(self, section: str, step: str) -> List[str]:
        """Returns 'techniques' list appropriate to 'step' in 'section'

        Args:
            section (str): name of section in 'contents' to search for
                techniques.
            step (str): name of 'step' for which techniques are sought.

        Returns:
            List[str]: techniques list stored in 'contents'.

        """
        return self._get_special(
                    section = section,
                    prefix = step,
                    suffix = 'techniques')

    def get_parameters(self, step: str, technique: str) -> Dict[str, Any]:
        """Returns 'parameters' dictionary appropriate to 'step' or 'technique'.

        The method firsts look for a match with 'technique' as a prefix (the
        more specific label) and then 'step' as a prefix' if there is no match
        for 'technique'.

        Args:
            step (str): name of 'step' for which parameters are sought.
            technique (str): name of 'technique' for which parameters are
                sought.

        Returns:
            Dict[str, Any]: parameters dictionary stored in 'contents'.

        """
        try:
            return self[f'{technique}_parameters']
        except KeyError:
            try:
                return self[f'{step}_parameters']
            except KeyError:
                raise KeyError(
                    f'parameters for {step} and {technique} not found')

    def inject(self,
            instance: object,
            overwrite: Optional[bool] = False) -> object:
        """Injects appropriate items into 'instance' from 'contents'.

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
            sections.extend(utilities.listify(instance.idea_sections))
        except AttributeError:
            pass
        for section in sections:
            try:
                for key, value in self.contents[section].items():
                    self._inject(
                        instance = instance,
                        attribute = key,
                        value = value,
                        overwrite = overwrite)
            except KeyError:
                pass
        return instance


class IdeaLoader(object):
    """Creates 'Idea' instance from file.

    Args:
        idea (Union[str, pathlib.Path, Dict[str, Any], 'Idea'])): a file path
            to a file with settings information, 'Idea'-compatible dictionary,
            or a completed 'Idea' instance.

    Returns:
        'Idea': properly configured.

    """
    idea: Union[str, pathlib.Path, Dict[str, Any], 'Idea']

    def __post_init__(self) -> 'Idea':
        """Creates an 'Idea' instance from the passed argument.

        Args:
            idea (Union[Dict[str, pathlib.Path, Dict[str, Any]], 'Idea']): a
                dict, a str file path to an .ini or .py file with settings, or
                an Idea instance with a 'contents' attribute.

        Returns:
            'Idea' instance, properly configured.

        Raises:
            TypeError: if 'idea' is neither a dict, str, Path, nor 'Idea'.

        """
        if isinstance(idea, Idea):
            return idea
        elif isinstance(idea, (dict, collections.abc.MutableMapping)):
            return Idea(contents = idea)
        if isinstance(idea, (str, pathlib.Path)):
            extension = str(pathlib.Path(idea).suffix)[1:]
            load_method = getattr(self, f'_load_from_{extension}')
            return Idea(contents = load_method(file_path = idea))
        elif idea is None:
            return Idea()
        else:
            raise TypeError('idea must be an Idea, str, Path, or dict type')

    def _load_from_ini(self, file_path: str) -> Dict[str, Any]:
        """Creates 'contents' from an .ini file.

        Args:
            file_path (str): path to configparser-compatible .ini file.

        Returns:
            Dict[str, Any] of contents.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a
                file.

        """
        try:
            contents = configparser.ConfigParser(dict_type = dict)
            contents.optionxform = lambda option: option
            contents.read(str(file_path))
            return dict(contents._sections)
        except FileNotFoundError:
            raise FileNotFoundError(' '.join(['contents file',
                file_path, 'not found']))

    def _load_from_py(self, file_path: str) -> Dict[str, Any]:
        """Creates 'contents' from a .py file.

        Args:
            file_path (str): path to python module with '__dict__' dict
                defined.

        Returns:
            Dict[str, Any] of contents.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a
                file.

        """
        try:
            return getattr(importlib.import_module(file_path), '__dict__')
        except FileNotFoundError:
            raise FileNotFoundError(' '.join(['contents file',
                file_path, 'not found']))