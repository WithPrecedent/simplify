"""
.. module:: configuration
:synopsis: configuration made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections
import collections.abc
import configparser
import dataclasses
import importlib
import pathlib
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import simplify
from simplify import core
from simplify.core import utilities


@dataclasses.dataclass
class SimpleSettings(collections.abc.MutableMapping):
    """Stores a siMpLify project settings.

    To create SimpleSettings instance, a user can pass a:
        1) file path to a compatible file type;
        2) string containing a a file path to a compatible file type;
                                or,
        3) dictionary.

    If 'contents' is imported from a file, 'SimpleSettings' creates a dictionary
    and can convert the dictionary values to appropriate datatypes. Currently
    supported file types are: .ini and .py. With .py files, users are free to
    set any datatypes in the original python file.

    For .ini files, SimpleSettings uses python's ConfigParser. It seeks to cure
    some of the shortcomings of the base ConfigParser including:
        1) All values in ConfigParser are strings.
        2) It does not automatically create a regular dictionary or compatiable
            MutableMapping.
        3) Access methods are unforgiving across the nested structure.
        4) It uses OrderedDict (python 3.6+ dictionaries are fast and ordered).

    If 'infer_types' is set to True (the default option), the dictionary values
    are automatically converted to appropriate datatypes (str, list, float,
    bool, and int are currently supported)

    Because SimpleSettings uses ConfigParser for .ini files, it only allows
    1- or 2-level settings dictionaries. The desire for accessibility and
    simplicity dictated this limitation. Further levels of nesting are not
    prohibited, but the forgiving '__getitem__' method will only catch for
    matches in the first nested level.

    Args:
        contents (Optional[Union[str, pathlib.Path, Dict[str, Any]]): a dict, a
            str file path to a file with settings, or a pathlib Path to a file
            with settings. Defaults to an empty dictionary.
        infer_types (Optional[bool]): whether values in 'contents' are converted
            to other datatypes (True) or left alone (False). If 'contents' was
            imported from an .ini file, a False value will leave all values as
            strings. Defaults to True.

    """
    contents: Optional[Union[
        str,
        pathlib.Path,
        Dict[str, Any]]] = dataclasses.field(default_factory = dict)
    infer_types: Optional[bool] = True

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Configures 'contents' to proper type.
        self.contents = self._validate_contents(contents = self.contents)
        # Infers types for values in 'contents', if option selected.
        if self.infer_types:
            self._infer_types()
        # Adds default settings as backup settings to 'contents'.
        self._chain_defaults()
        return self

    """ Core siMpLify Methods """

    def add(self, section: str, settings: Dict[str, Any]) -> None:
        """Adds 'settings' to 'contents'.

        Args:
            section (str): name of section to add 'dictionary' to.
            settings (Dict[str, Any]): dict to add to 'section'.

        """
        if section in self:
            self[section].update(settings)
        else:
            self[section] = settings
        return self

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
            sections.extend(utilities.listify(instance.settings_sections))
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
            raise KeyError(f'{key} is not found in {self.__class__.__name__}')

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

    def _validate_contents(self, contents: Union[
            str,
            pathlib.Path,
            Dict[str, Any]]) -> Dict[str, Any]:
        """

        Args:
            contents (Union[str,pathlib.Path,Dict[str, Any]]): [description]

        Raises:
            TypeError: [description]

        Returns:
            Dict[str, Any]: [description]

        """
        if isinstance(contents, (dict, collections.abc.MutableMapping)):
            return contents
        elif isinstance(contents, (str, pathlib.Path)):
            extension = str(pathlib.Path(contents).suffix)[1:]
            load_method = getattr(self, f'_load_from_{extension}')
            return load_method(file_path = contents)
        elif contents is None:
            return {}
        else:
            raise TypeError('contents must be a dict, Path, or str type')

    def _load_from_ini(self, file_path: str) -> Dict[str, Any]:
        """Returns settings dictionary from an .ini file.

        Args:
            file_path (str): path to configparser-compatible .ini file.

        Returns:
            Dict[str, Any] of contents.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            contents = configparser.ConfigParser(dict_type = dict)
            contents.optionxform = lambda option: option
            contents.read(str(file_path))
            return dict(contents._sections)
        except FileNotFoundError:
            raise FileNotFoundError(f'settings file {file_path} not found')

    def _load_from_py(self, file_path: str) -> Dict[str, Any]:
        """Returns a settings dictionary from a .py file.

        Args:
            file_path (str): path to python module with '__dict__' dict
                defined.

        Returns:
            Dict[str, Any] of contents.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a
                file.

        """
        # Disables type conversion if the source is a python file.
        self.infer_types = False
        try:
            return getattr(importlib.import_module(file_path), '__dict__')
        except FileNotFoundError:
            raise FileNotFoundError(f'settings file {file_path} not found')

    def _infer_types(self,
            contents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Converts stored values to appropriate datatypes.

        Args:
            contents (Dict[str, Dict[str, Any]]): a nested contents dictionary
                to review.

        Returns:
            Dict[str, Dict[str, Any]]: with the nested values converted to the
                appropriate datatypes.

        """
        new_contents = {}
        for key, value in contents.items():
            if isinstance(value, dict):
                inner_bundle = {
                    inner_key: utilities.typify(inner_value)
                    for inner_key, inner_value in value.items()}
                new_contents[key] = inner_bundle
            else:
                new_contents[key] = utilities.typify(value)
        return new_contents

    def _chain_defaults(self) -> None:
        """Creates set of mappings for siMpLify settings lookup."""
        defaults = {k.lower(): v for k, v in core.defaults.__dict__.items()}
        self.contents = collections.ChainMap(self.contents, defaults)
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
        return instanceH