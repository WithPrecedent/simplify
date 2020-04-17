"""
.. module:: plan
:synopsis: siMpLify project iterable
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import simplify
from simplify import core


@dataclasses.dataclass
class SimplePlan(core.SimpleComponent, collections.abc.MutableMapping):
    """Base class for iterating SimpleComponent instances.

    A SimplePlan stores a list of items with 'name' attributes. Each 'name'
    acts as a key to create the facade of a dictionary with the items in the
    stored list serving as values. This allows for duplicate keys and the
    storage of class instances at the expense of lookup speed. Since normal
    use cases do not include repeat accessing of SimplePlan instances, the
    loss of lookup speed should have negligible effect on overall performance.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared Idea instance, 'name'
            should match the appropriate section name in that Idea instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        contents (Optional[List[SimpleComponent]]): stored list. Defaults to an
            empty list.

    """
    name: Optional[str] = None
    contents: Optional[List[core.SimpleComponent]] = dataclasses.field(
        default_factory = list)

    """ Public Methods """

    def add(self,
            contents: Union[
                List[core.SimpleComponent],
                core.SimpleComponent]) -> None:
        """Adds argument with 'contents'.

        If a list is passed, contents is extended to include it. However, if a
        SimpleComponent is passed, contents is appended with it.

        Args:
            contents (Union[List[SimpleComponent], SimpleComponent]):
                SimpleComponent(s) to add to 'contents'.

        Raises:
            TypeError: if 'contents' is not a list or SimpleComponent

        """
        if isinstance(contents, core.SimpleComponent):
            self.contents.append(contents)
        elif isinstance(contents, list):
            self.contents.extend(contents)
        else:
            raise TypeError(f'contents must be a SimpleComponent or list type')
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> List[core.SimpleComponent]:
        """Returns value(s) for 'key' in 'contents' as a list.

        Args:
            key (str): name to search for in 'contents'.

        Returns:
            List[SimpleComponent]: value(s) stored in 'contents'.

        """
        return [c for c in self.contents if c.name == key]

    def __setitem__(self, key: str, value: core.SimpleComponent) -> None:
        """Adds 'value' to 'contents' if 'key' matches 'value.name'.

        Args:
            key (str): name of key(s) to set in 'contents'.
            value (SimpleComponent): value(s) to be added at the end of
                'contents'.

        Raises:
            TypeError: if 'name' attribute in value either doesn't exist or
                doesn't match 'key'.

        """
        if hasattr(value, name) and value.name in [key]:
            self.add(contents = contents)
        else:
            raise TypeError(
                f'{self.name} requires a value with a name atttribute')
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (str): name(s) of key(s) in 'contents' to
                delete the key/value pair.

        """
        try:
            self.contents = [item for item in self.contents if item.name != key]
        except AttributeError:
            raise TypeError(
                f'{self.name} requires a value with a name atttribute')
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
            other: Union[
                List[core.SimpleComponent],
                core.SimpleComponent]) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union[List[SimpleComponent], SimpleComponent]):
                SimpleComponent(s) to add to 'contents'.

        """
        self.add(contents = other)
        return self

    def __iadd__(self,
            other: Union[
                List[core.SimpleComponent],
                core.SimpleComponent]) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union[List[SimpleComponent], SimpleComponent]):
                SimpleComponent(s) to add to 'contents'.

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
        """Returns representation of 'contents'.

        Returns:
            str: representation of 'contents'.

        """
        return (
            f'siMpLify {self.name} '
            f'contents: {self.contents} ')