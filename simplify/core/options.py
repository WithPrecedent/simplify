"""
.. module:: options
:synopsis: base class for containing different options
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify


@dataclass
class SimpleOptions(MutableMapping):
    """Base class for different options to be stored.

    The SimpleOptions class should be injected with the shared Idea instance
    before it or any subclass is instanced. This is done automatically through
    the normal siMpLify access points. But if creating a completely customized
    workflow, this step must be taken for siMpLify to work properly.

    Args:
        options (Optional[Dict[str, Any]]): alternative strategies stored
            in a dictionary in the following format:

                {str: Outline}

            If subclassing, 'drafted' should be declared in the 'draft' method.
            Defaults to an empty dict.
        default_options (Optional[Union[List[str], str]]): key(s) to use if
            'default' is selected. Defaults to an empty list. If not specified,
            and 'default' options are sought, all options will be returned.

    """
    options: Optional[Dict[str, Any]] = field(default_factory = dict())
    default_options: Optional[Union[List[str], str]] = field(
        default_factory = list())

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        # Sets wildcard values to check if a key doesn't exist in options.
        self.wildcards = {
            'all': self.all,
            'default': self.default,
            'defaults': self.default,
            'none': ['none'],
            'None': ['none']}
        # Initializes state-dependent dictionaries.
        self.drafted = {}
        self.published = {}
        self.applied = {}
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in options.

        Args:
            item (str): name of key in options.

        """
        try:
            del getattr(self, self.state)[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> Any:
        """Returns item in options.

        If there are no matches, the method searches for a matching wildcard.

        Args:
            item (str): name of key in options.

        Raises:
            KeyError: if 'item' is not found in options and does not match
                a recognized wildcard.

        """
        try:
            return getattr(self, self.state)[item]
        except KeyError:
            try:
                return self.wildcards[item]
            except KeyError:
                raise KeyError(' '.join([item, 'is not in', self.name]))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets 'item' in options to 'value'.

        Args:
            item (str): name of key in options.
            value (Any): value to be paired with 'item' in options.

        """
        getattr(self, self.state)[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of options."""
        return iter(getattr(self, self.state))

    def __len__(self) -> int:
        """Returns length of options."""
        return len(getattr(self, self.state))

    """ Numeric Dunder Methods """

    def __add__(self, other: Union[Dict[str, Any], 'SimpleOptions']) -> None:
        """Combines two options dictionaries.

        Args:
            other (Union[Dict[str, Any],): either another 'SimpleOptions'
                instance or an options dict.

        Raises:
            TypeError: if 'other' is neither a 'SimpleOptions' instance nor
                a dict.

        """
        try:
            getattr(self, self.state).update(getattr(other, self.state))
        except AttributeError:
            try:
                getattr(self, self.state).update(other)
            except AttributeError:
                raise TypeError(' '.join(
                    ['addition requires objects to be dict or SimpleOptions']))
        return self

    def __iadd__(self, other: Union[Dict[str, Any], 'SimpleOptions']) -> None:
        """Combines two options dictionaries.

        Args:
            other (Union[Dict[str, Any],): either another 'SimpleOptions'
                instance or an options dict.

        Raises:
            TypeError: if 'other' is neither a 'SimpleOptions' instance nor
                a dict.

        """
        self.__add__(other = other)
        return self

    def __invert__(self) -> None:
        """Reverses keys and values in options."""
        try:
            setattr(self, self.state, self.__reversed__())
        except AttributeError:
            setattr(self, self.state, {})
        return self

    """ Sequence Dunder Methods """

    def __reversed__(self) -> Dict[Any, str]:
        """Returns options with keys and values reversed."""
        return {value: key for key, value in getattr(self, self.state).items()}

    """ Core siMpLify Methods """

    def load(self, keys: Optional[Union[str, List[str]]] = None) -> object:
        """Returns object from module based upon tuple in options value.

        Args:
            keys (Optional[Union[str, List[str]]]): key(s) of option(s) to be
                loaded. Defaults to None. If not provided, all options will be
                loaded.

        """
        if keys is None:
            keys = list(self.options.keys())
        for key in listify(keys):
            self.published[key] = getattr(self, self.state)[key].load()
        return self

    def draft(self) -> None:
        """Subclasses should call super().draft() and declare 'drafted' here.

        Also, if any default_options are to be set independent of the instance
        arguments, that should be done here as well.

        If the default 'wildcards' attribute is to be overrided, this should be
        done in this method by any subclass.

        """
        # Sets state for access methods.
        self.state = 'drafted'
        # Assigns initial options in a 'drafted' state.
        self.drafted = self.options
        return self

    def publish(self, data: Optional[object] = None) -> None:
        """"Finalizes options.

        Args:
            data (Optional[object]): an object to pass when an options instance
                is created. Defaults to None.

        """
        # Sets state for access methods.
        self.state = 'published'
        # Sets 'default_options' to all options if none exist.
        if not self.default_options:
            self.default_options = self.all
        # Lazily loads all stored options from stored Outline instances.
        self.options.load()
        # Instances and publishes all selected options.
        for key, option in self.options.items():
            try:
                instance = option(idea = self.idea)
                instance.publish(data = data)
                self.published[key] = instance
            except AttributeError:
                pass
        return self

    def apply(self, key: str, **kwargs) -> 'Simple_Manuscript':
        # Sets state for access methods.
        self.state = 'applied'
        try:
            return self.applied[key]
        except (AttributeError, KeyError):
            self.applied[key] = self.published[key](**kwargs)
            return self.applied[key]

    """ Properties """

    @property
    def all(self):
        return list(self.drafted.keys())

    @property
    def default(self):
        return self.default_options

    @default.setter
    def default(self,
            options: Union[str, List[str]],
            override: Optional[bool]) -> None:
        if override or not self.default_options:
            default_options = listify(options)
        else:
            default_options.extend(listify(options))
        return self
