"""
.. module:: repository
:synopsis: siMpLify base mapping classes.
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify
from simplify.core.utilities import subsetify


@dataclass
class Repository(MutableMapping):
    """A flexible dictionary that includes wildcard keys.

    The base class includes 'default', 'all', and 'none' wilcard properties
    which can be accessed through dict methods by those names. Users can also
    set the 'default' and 'none' properties to change what is returned when the
    corresponding keys are sought.

    Args:
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        wildcards (Optional[List[str]]): a list of corresponding properties
            which access sets of dictionary keys. If none is passed, the two
            included properties ('default' and 'all') are used.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        null_value (Optional[Any]): value to return when 'none' is accessed or
            an item isn't found in 'contents'. Defaults to None.

    """
    contents: Optional[Dict[str, Any]] = field(default_factory = dict)
    wildcards: Optional[List[str]] = field(default_factory = list)
    defaults: Optional[List[str]] = field(default_factory = list)
    null_value: Optional[Any] = None

    def __post_init__(self) -> None:
        """Initializes 'defaults' and 'wildcards'."""
        if not self.wildcards:
            self.wildcards = ['all', 'default', 'none']
        if not self.defaults:
            self.defaults = list(self.contents.keys())
        if hasattr(self, '_create_contents'):
            self._create_contents()
        self.nestify()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns value for 'key' in 'contents'.

        If there are no matches, the method searches for a matching wildcard in
        attributes.

        Args:
            key (str): name of key in 'contents'.

        Returns:
            Any: item stored in 'contents' or 'wildcard'.

        """
        try:
            return self.contents[key]
        except KeyError:
            if key in self.wildcards:
                return getattr(self, key)
            else:
                raise KeyError(' '.join(
                    [key, 'not found in', self.__class__.__name__]))

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

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (str): name of key in 'contents'.
            value (Any): value to be paired with 'key' in 'contents'.

        """
        self.contents[key] = value
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

    """ Other Dunder Methods """

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.contents.__str__()

    """ Public Methods """

    def nestify(self) -> None:
        """Converts 1 level of nested dictionaries to Repository instances."""
        for key, value in self.contents.items():
            if isinstance(value, dict):
                self.contents[key] = Repository(contents = value)
        return self

    """ Wildcard Properties """

    @property
    def all(self) -> Dict[str, Any]:
        """Returns 'contents' values.

        Returns:
            List[str] of values stored in 'contents'.

        """
        return self.contents

    @property
    def default(self) -> Dict[str, Any]:
        """Returns key/values for keys in '_default'.

        Returns:
            List[str]: keys stored in 'defaults' of 'contents'.

        """
        try:
            return subsetify(self.contents, self._default)
        except AttributeError:
            self._default = self.all
            return self.all

    @default.setter
    def default(self, keys: Union[List[str], str]) -> None:
        """Sets '_default' to 'contents'

        Args:
            keys (Union[List[str], str]): list of keys in 'contents' to return
                when 'default' is accessed.

        """
        self._default = listify(keys)
        return self

    @default.deleter
    def default(self, keys: Union[List[str], str]) -> None:
        """Removes 'contents' from '_default'.

        Args:
            keys (Union[List[str], str]): list of keys in 'contents' to remove
                from '_default'.

        """
        for option in listify(keys):
            try:
                del self._default[option]
            except KeyError:
                pass
        return self

    @property
    def none(self) -> Any:
        """Returns 'null_value'.

        Returns:
            Any: 'null_value' attribute.

        """
        return self.null_value

    @default.setter
    def none(self, null_value: Any) -> None:
        """Sets 'none' to 'null_value'.

        Args:
            null_value (Any): value to return when 'none' is sought.

        """
        self.null_value = null_value
        return self


@dataclass
class Sequence(Repository):
    """A flexible dictionary that keeps wildcards and a separate ordered list.

    Args:
        contents (Optional[str, Any]): default stored dictionary. Defaults to
            an empty dictionary.
        wildcards (Optional[List[str]]): a list of corresponding properties
            which access sets of dictionary keys. If none is passed, the two
            included properties ('default' and 'all') are used.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        null_value (Optional[Any]): value to return when 'none' is accessed or
            an item isn't found in 'contents'. Defaults to None.
        order (Optional[List[str]]): the order the keys in 'contents' should
            be accessed. Even though python (3.7+) are now ordered, the order
            is dependent upon when an item is added. This attribute allows
            the dictionary to be iterated based upon a separate variable which
            can be updated with the 'order' property. If none is passed, the
            initial order of the keys in 'contents' is used.

    """
    contents: Optional[Dict[str, Any]] = field(default_factory = dict)
    wildcards: Optional[List[str]] = field(default_factory = list)
    defaults: Optional[List[str]] = field(default_factory = list)
    null_value: Optional[Any] = None
    order: Optional[List[str]] = field(default_factory = list)

    def __post_init__(self) -> None:
        """Initializes '_order', 'defaults', and 'wildcards'."""
        if self.contents and not self.order:
            self._order = list(self.contents.keys())
        super().__post_init__()
        return self

    """ Required ABC Methods """

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'contents'.

        Args:
            key (str): name of key in 'contents'.

        """
        try:
            del self.contents[key]
            self.order.remove[key]
        except KeyError:
            pass
        return self

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (str): name of key in 'contents'.
            value (Any): value to be paired with 'key' in 'contents'.

        """
        self.contents[key] = value
        self.order.append(key)
        return self

    """ Other Dunder Methods """

    def __add__(self, other: Union['Repository', Dict[str, Any]]) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union['Repository', Dict[str, Any]]): another
                'Repository' instance or compatible dictionary.

        """
        self.add(contents = other)
        return self

    def __iadd__(self, other: Union['Repository', Dict[str, Any]]) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union['Repository', Dict[str, Any]]): another
                'Repository' instance or compatible dictionary.

        """
        self.add(contents = other)
        return self

    """ Public Methods """

    def add(self,
            key: Optional[str] = None,
            value: Optional[Any] = None,
            contents: Optional[Union[
                'Repository', Dict[str, Any]]] = None) -> None:
        """Combines arguments with 'contents'.

        Args:
            key (Optional[str]): key for 'value' to use. Defaults to None.
            value (Optional[Any]): item to store in 'contents'. Defaults to
                None.
            contents (Optional[Union['Repository', Dict[str, Any]]]):
                another 'Repository' instance/subclass or a compatible
                dictionary. Defaults to None.

        """
        if key is not None and value is not None:
            self.contents[key] = value
            self.order.append(key)
        if contents is not None:
            self.update(contents = contents)
        return self

    """ Dictionary Compatibility Methods """

    def update(self,
            contents: Union['Repository', Dict[str, Any]] = None) -> None:
        """Combines argument with 'contents'.

        Args:
            contents ([Union['Repository', Dict[str, Any]]): another
                'Repository' instance/subclass or a compatible dictionary.
                Defaults to None.

        """
        try:
            self.contents.update(getattr(contents, contents.dictionary))
            self.order.extend(getattr(contents, contents.order))
        except AttributeError:
            try:
                self.contents.update(contents)
                self.order.extend(list(contents.keys()))
            except (TypeError, AttributeError):
                raise TypeError('contents must be dict, Sequence, or Repository')
        return self

    """ Order Property """

    @property
    def order(self) -> List[str]:
        """Returns '_order' or list of keys of 'contents'.

        Returns:
            List[str]: keys stored in '_order' of 'contents'.

        """
        try:
            self._order = deduplicate(
                [x for x in self._order if x in self.contents.keys()])
            return self._order
        except AttributeError:
            self._order = list(self.contents.keys())
            return self._order

    @order.setter
    def order(self, keys: Union[List[str], str]) -> None:
        """Sets '_order' to 'contents'

        Args:
            keys (Union[List[str], str]): list of keys in 'contents' to return
                when '_order' is accessed.

        """
        self._order = listify(keys)
        return self

    @order.deleter
    def order(self, keys: Union[List[str], str]) -> None:
        """Removes 'contents' from '_order'.

        Args:
            keys (Union[List[str], str]): list of keys in 'contents' to remove
                from '_order'.

        """
        for item in listify(keys):
            try:
                self._order.remove(item)
            except KeyError:
                pass
            except AttributeError:
                self._order = list(self.contents.keys())
                self._order.remove(item)
        return self
