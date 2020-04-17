"""
.. module:: repository
:synopsis: project option storage made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import simplify
from simplify import core
from simplify.core import utilities


@dataclasses.dataclass
class SimpleRepository(core.SimpleComponent, collections.abc.MutableMapping):
    """Base class for policy and option storage.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared Idea instance, 'name'
            should match the appropriate section name in that Idea instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.

    """
    name: Optional[str] = None
    contents: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory = dict)
    defaults: Optional[List[str]] = dataclasses.field(default_factory = list)

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super().__post_init__()
        # Stores nested dictionaries as SimpleRepository instances.
        self.contents = self._nestify(contents = self.contents)
        # Sets 'default' to all keys of 'contents', if not passed.
        self.defaults = self.defaults or list(self.contents.keys())
        return self

    """ Public Methods """

    def add(self, contents: Union['SimpleRepository', Dict[str, Any]]) -> None:
        """Combines arguments with 'contents'.

        Args:
            contents (Union[SimpleRepository, Dict[str, Any]]): another
                SimpleRepository instance/subclass or a compatible dictionary.

        """
        self.contents.update(contents)
        self.contents = self._nestify(contents = self.contents)
        return self

    def subset(self, subset: Union[Any, List[Any]]) -> 'SimpleRepository':
        """Returns a subset of 'contents'.

        Args:
            subset (Union[Any, List[Any]]): key(s) to get key/value pairs from
                'dictionary'.

        Returns:
            'SimpleRepository': with only keys in 'subset'.

        """
        return self.__class__(
            name = name,
            contents = utilities.subsetify(
                dictionary = self.contents,
                subset = subset),
            defaults = self.defaults)

    """ Required ABC Methods """

    def __getitem__(self, key: Union[List[str], str]) -> Union[List[Any], Any]:
        """Returns value(s) for 'key' in 'contents'.

        The method searches for 'all', 'default', and 'none' matching wildcard
        options before searching for direct matches in 'contents'.

        Args:
            key (Union[List[str], str]): name(s) of key(s) in 'contents'.

        Returns:
            Union[List[Any], Any]: value(s) stored in 'contents'.

        """
        if key in ['all', ['all']]:
            return list(self.contents.values())
        elif key in ['default', ['default']]:
            return list(utilities.subsetify(
                dictionary = self.contents,
                subset = self.defaults).values())
        elif key in ['none', ['none'], '', ['']]:
            return []
        else:
            try:
                return self.contents[key]
            except TypeError:
                try:
                    return [self.contents[k] for k in key if k in self.contents]
                except KeyError:
                    raise KeyError(f'{key} is not in {self.name}')
            except KeyError:
                raise KeyError(f'{key} is not in {self.name}')

    def __setitem__(self,
            key: Union[List[str], str],
            value: Union[List[Any], Any]) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (Union[List[str], str]): name of key(s) to set in 'contents'.
            value (Union[List[Any], Any]): value(s) to be paired with 'key' in
                'contents'.

        """
        if key in ['default', ['default']]:
            self.defaults = value
        else:
            try:
                self.contents[key] = value
            except TypeError:
                self.contents.update(dict(zip(key, value)))
        return self

    def __delitem__(self, key: Union[List[str], str]) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (Union[List[str], str]): name(s) of key(s) in 'contents' to
                delete the key/value pair.

        """
        self.contents = {
            i: self.contents[i]
            for i in self.contents if i not in utilities.listify(key)}
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'contents'.

        Returns:
            Iterable: of 'contents'.

        """
        return iter(self.contents)

    def __len__(self) -> int:
        """Returns length of 'contents'.

        Returns:
            Integer: length of 'contents'.

        """
        return len(self.contents)

    """ Other Dunder Methods """

    def __add__(self,
            other: Union['SimpleRepository', Dict[str, Any]]) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union['SimpleRepository', Dict[str, Any]]): another
                'SimpleRepository' instance or compatible dictionary.

        """
        self.add(contents = other)
        return self

    def __iadd__(self,
            other: Union['SimpleRepository', Dict[str, Any]]) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union['SimpleRepository', Dict[str, Any]]): another
                'SimpleRepository' instance or compatible dictionary.

        """
        self.add(contents = other)
        return self

    def __repr__(self) -> str:
        """Returns '__str__' representation.

        Returns:
            str: default dictionary representation of 'contents'.

        """
        return self.__str__()

    def __str__(self) -> str:
        """Returns default dictionary representation of contents.

        Returns:
            str: default dictionary representation of 'contents'.

        """
        return (
            f'{self.name} '
            f'contents: {self.contents.__str__()} '
            f'defaults: {self.defaults} ')

    """ Private Methods """

    def _nestify(self,
            contents: Union[
                'SimpleRepository',
                Dict[str, Any]]) -> 'SimpleRepository':
        """Converts nested dictionaries to 'SimpleRepository' instances.

        Args:
            contents (Union['SimpleRepository', Dict[str, Any]]): mutable
                mapping to be converted to a 'SimpleRepository' instance.

        Returns:
            'SimpleRepository': subclass instance with 'contents' stored.

        """
        new_repository = self.__new__()
        for key, value in contents.items():
            if isinstance(value, dict):
                new_repository.add(
                    contents = {key: self._nestify(contents = value)})
            else:
                new_repository.add(contents = {key: value})
        return new_repository