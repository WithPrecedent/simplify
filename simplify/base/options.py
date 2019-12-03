"""
.. module:: options
:synopsis: base class and mixin for options
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify


@dataclass
class SimpleOptions(MutableMapping):
    """Base class or mixin for classes with 'options' attribute."""

    def __post_init__(self):
        self._draft_options()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in 'options' or instance attribute.

        The method looks in 'options' first, but, if 'attribute' is not found
        there, it looks for an instance attribute. If neither is found, the
        method takes no further action.

        Args:
            attribute (str): name of key in 'options' or instance attribute.

        """
        try:
            del self.options[item]
        except KeyError:
            try:
                del self.__dict__[item]
            except AttributeError:
                pass
        except AttributeError:
            self.options = {}
        return self

    def __getitem__(self, item: str) -> Any:
        """Returns item in 'options' or instance attribute.

        Args:
            attribute (str): name of key in 'options' or instance attribute.

        Returns:
            Any: value in 'options' or, if 'attribute' is not found there, an
                instance attribute.

        Raises:
            AttributeError: if 'attribute' is neither in 'options' nor an
                instance attribute.

        """
        try:
            return self.options[item]
        except KeyError:
            try:
                return self.__dict__[item]
            except AttributeError:
                try:
                    raise KeyError(' '.join([item, 'is not in', self.name]))
                except AttributeError:
                    raise KeyError(' '.join(
                        [item, 'is not in', self.__class__.__name__]))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets item in 'options' to 'value'.

        Args:
            attribute (str): name of key in 'options' or instance attribute.
            value (Any): value to be paired with 'attribute' in 'options'.

        """
        try:
            self.options[item] = value
        except AttributeError:
            self.options = {item: value}
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'options'."""
        try:
            return iter(self.options)
        except AttributeError:
            self.options = {}
            return iter(self.options)

    def __len__(self) -> int:
        """Returns length of 'options'."""
        try:
            return len(self.options)
        except AttributeError:
            self.options = {}
            return len(self.options)

    """ Numeric Dunder Methods """

    def __add__(self, other: Union[Dict[str, Any], 'SimpleOptions']) -> None:
        """Combines two 'options' dictionaries.

        Args:
            other (Union[Dict[str, Any],): either another 'SimpleOptions'
                instance or an 'options' dict.

        Raises:
            TypeError: if 'other' is neither a 'SimpleOptions' instance nor
                dict.

        """
        try:
            self.options.update(other.options)
        except AttributeError:
            try:
                self.options.update(other)
            except AttributeError:
                try:
                    self.options = other.options
                except AttributeError:
                    if isinstance(other, dict):
                        self.options = other
                    else:
                        raise TypeError(' '.join(
                            ['addition requires both objects be dict or',
                            'SimpleOptions']))
        return self

    def __iadd__(self, other: Union[Dict[str, Any], 'SimpleOptions']) -> None:
        """Combines two 'options' dictionaries.

        Args:
            other (Union[Dict[str, Any],): either another 'SimpleOptions'
                instance or an 'options' dict.

        Raises:
            TypeError: if 'other' is neither a 'SimpleOptions' instance nor
                dict.

        """
        self.__add__(other = other)
        return self

    def __invert__(self) -> None:
        """Reverses keys and values in 'options'."""
        try:
            reversed = self.__reversed__()
            self.options = reversed
        except AttributeError:
            self.options = {}
        return self

    """ Sequence Dunder Methods """

    def __reversed__(self) -> Dict[Any, str]:
        """Returns 'options' with keys and values reversed."""
        try:
            return {value: key for key, value in self.options.items()}
        except AttributeError:
            self.options = {}
            return {}

    """ Private Methods """

    def _convert_wildcards(self, value: Union[str, List[str]]) -> List[str]:
        """Converts 'all', 'default', or 'none' values to a list of items.

        Args:
            value (Union[str, List[str]]): name(s) of pages.

        Returns:
            If 'all', either the 'all' property or all keys listed in 'options'
                dictionary are returned.
            If 'default', either the 'defaults' property or all keys listed in
                'options' dictionary are returned.
            If some variation of 'none', 'none' is returned.
            Otherwise, 'value' is returned intact.

        """
        if value in ['all', ['all']]:
            return self.all
        elif value in ['default', ['default']]:
            self.default
        elif value in ['none', ['none'], 'None', ['None'], None]:
            return ['none']
        else:
            return listify(value)

    def _draft_options(self) -> None:
        """Declares 'options' dict.

        Subclasses should provide their own '_draft_options' method, if needed.

        """
        if not hasattr(self, 'options') or not self.options:
            self.options = {}
        return self

    """ Public Methods """

    def add_options(self, options: Dict[str, Any]) -> None:
        """Adds new 'options' to class instance 'options' attribute.

        Args:
            options (Dict[str, Any]): options to be added.

        """
        try:
            self.options.update(options)
        except AttributeError:
            self.options = options
        return self
