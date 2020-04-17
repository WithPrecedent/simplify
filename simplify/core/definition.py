"""
.. module:: definitions
:synopsis: typing made simple
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
class SimpleType(collections.abc.MutableMapping):
    """Base class for proxy typing.

    Args:
        types (Dict[str, Any]): keys are proxy names of types and values are
            the actual types.

    """
    types: Dict[str, Any]

    def __post_init__(self) -> None:
        """Creates 'reversed_types' from passed 'types'."""
        self._create_reversed()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns 'key' in the 'types' or 'reversed_types' dictionary.

        Args:
            key (str): name of key to find.

        Returns:
            Any: value stored in 'types' or 'reversed_types' dictionaries.

        Raises:
            KeyError: if 'key' is neither found in 'types' nor 'reversed_types'
                dictionaries.

        """
        try:
            return self.types[key]
        except KeyError:
            try:
                return self.reversed_types[key]
            except KeyError:
                raise KeyError(f'{key} is not in {self.__class__.__name__}')

    def __setitem__(self, key: str, value: Any) -> None:
        """Stores arguments in 'types' and 'reversed_types' dictionaries.

        Args:
            key (str): name of key to set.
            value (Any): value tto be paired with key.

        """
        self.types[key] = value
        self.reversed_types[value] = key
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes key in the 'types' and 'reversed_types' dictionaries.

        Args:
            key (str): name of key to delete.

        """
        try:
            value = self.types[key]
            del self.types[key]
            del self.reversed_types[value]
        except KeyError:
            try:
                value = self.reversed_types[key]
                del self.reversed_types[key]
                del self.types[value]
            except KeyError:
                pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of the 'types' dictionary.

        Returns:
            Iterable stored in the 'types' dictionary.

        """
        return iter(self.types)

    def __len__(self) -> int:
        """Returns length of the 'types' dictionary if 'iterable' not set..

        Returns:
            int of length of 'types' dictionary.

        """
        return len(self.types)

    """ Private Methods """

    def _create_reversed(self) -> None:
        """Creates 'reversed_types'."""
        self.reversed_types = {value: key for key, value in self.types.items()}
        return self