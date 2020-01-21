"""
.. module:: types
:synopsis: siMpLify base class definitions and types
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from collections.abc import Iterable
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simplify.core.utilities import listify


@dataclass
class Outline(Container):
    """Object construction instructions used by Editor subclasses.

    Ideally, this class should have no additional methods beyond the lazy
    loader ('load' method) and __contains__  dunder method.

    Users can use the idiom 'x in Option' to check if a particular attribute
    exists and is not None. This means default values for optional arguments
    should generally be set to None to allow use of that idiom.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify module).
        default_module (Optional[str]): a backup module location if a component
            is not found in 'module'. Defaults to None. If not provided,
            siMpLify uses 'simplify.core' as 'default_module'.

    """
    name: str
    module: str
    default_module: Optional[str] = None

    def __post_init__(self) -> None:
        """Sets 'default_module' if none is provided."""
        if self.default_module is None:
            self.default_module = 'simplify.core'
        return self

    """ Required ABC Methods """

    def __contains__(self, attribute: str) -> bool:
        """Returns whether attribute exists in a subclass instance.

        Args:
            attribute (str): name of attribute to check.

        Returns:
            bool: whether the attribute exists and is not None.

        """
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    """ Public Methods """

    def load(self, component: str) -> object:
        """Returns 'component' from 'module'.

        Args:
            component (str): name of object to load from 'module'.

        Returns:
            object: from 'module'.

        """
        try:
            return getattr(
                import_module(self.module),
                getattr(self, component))
        except (ImportError, AttributeError):
            try:
                return getattr(
                    import_module(self.default_module),
                    getattr(self, component))
            except (ImportError, AttributeError):
                raise ImportError(' '.join(
                    [component, 'is neither in', self.module, 'nor',
                        self.default_module]))


@dataclass
class SimpleType(MutableMapping):
    """Base class for proxy typing."""

    types: Dict[str, Any]

    def __post_init__(self) -> None:
        """Creates 'reversed_types' from passed 'types'."""
        self._create_reversed()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns key in the 'types' or 'reversed_types' dictionary.

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
                raise KeyError(' '.join(
                    [key, 'is not in', self.__class__.__name__]))

    def __setitem__(self, key: str, value: Any) -> None:
        """Stoes arguments in 'types' and 'reversed_types' dictionaries.

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
            key (str): name of key to find.

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


@dataclass
class SimpleDatatypes(SimpleType):

    def __post_init__(self) -> None:
        self.datatypes = {
            'boolean': bool,
            'float': float,
            'integer': int,
            'string': object,
            'categorical': pd.CategoricalDtype,
            'list': list,
            'datetime': np.datetime64,
            'timedelta': datetime.timedelta}
        return self