"""
.. module:: overview
:synopsis: siMpLify project overview
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)
import warnings


@dataclass
class Overview(MutableMapping):
    """Stores outline of a siMpLify project.

    Args:
        contents (Optional[Dict[str, Dict[str, List[str]]]]): dictionary
            storing the outline of a siMpLify project. Defaults to an empty
            dictionary.


    """
    contents: Optional[Dict[str, Dict[str, List[str]]]] = field(
        default_factory = dict)

    """ Factory Method """

    @classmethod
    def create(cls, manager: 'Manager') -> 'Overview':
        """Creates an 'Overview' instance from 'workers'.

        Args:
            manager ('Manager'): an instance with stored 'workers'.

        Returns:
            'Overview': instance, properly configured.

        """
        contents = {}
        for name, worker in manager.workers.items():
            contents[name] = worker.outline()
        return cls(contents = contents)

    """ Required ABC Methods """

    def __getitem__(self, 
            key: Union[str, Tuple[str, str]]) -> Union[
                Dict[str, List[str]], List[str]]:
        """Returns key from 'contents'.

        Args:
            key (Union[str, Tuple[str, str]]): key to item in 'contents'. If
                'key' is a tuple, the method attempts to return [key[0]][key[1]]
                from 'contents'.

        Returns:
            Union[Dict[str, List[str]], List[str]]]:man overview of either 
                one package of a siMpLify project (if 'key' is a str) or one 
                step in one package of a siMpLify project (if 'key' is a tuple).

        Raises:
            TypeError: if 'key' is neither a str nor tuple type.

        """
        if isinstance(key, str):
            return self.contents[key]
        elif isinstance(key, tuple):
            return self.contents[key[0]][key[1]]
        else:
            raise TypeError(
                f'{self.__class__.__name__} requires str or tuple type')

    def __setitem__(self,
            key: Union[str, Tuple[str, str]],
            value: Union[Dict[str, List[str]], List[str]]) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (Union[str, Tuple[str, str]]): key to item in 'contents' to set.
            value (Union[Dict[str, List[str]], List[str]]): dictionary or list 
                to place in 'contents'.

        Raises:
            TypeError: if 'key' is neither a str nor tuple type.

        """
        if isinstance(key, str):
            self.contents[key] = value
        elif isinstance(key, tuple):
            if not key[0] in self.contents:
                self.contents[key[0]] = {}
            self.contents[key[0]][key[1]] = value
        else:
            raise TypeError(
                f'{self.__class__.__name__} requires str or tuple type')
        return self

    def __delitem__(self, key: Union[str, Tuple[str, str]]) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (Union[str, Tuple[str, str]]): key in 'contents'.

        Raises:
            TypeError: if 'key' is neither a str nor tuple type.

        """
        if isinstance(key, str):
            try:
                del self.contents[key]
            except KeyError:
                pass
        elif isinstance(key, tuple):
            try:
                del self.contents[key[0]][key[1]]
            except KeyError:
                pass
        else:
            raise TypeError(
                f'{self.__class__.__name__} requires str or tuple type')
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
            int: length of 'contents'.
        """
        return len(self.contents)

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return (f'Project {self.identification}:',
                f'{self.contents}')