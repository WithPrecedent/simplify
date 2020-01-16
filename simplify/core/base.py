"""
.. module:: base
:synopsis: siMpLify base classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from collections.abc import Collection
from collections.abc import Container
from collections.abc import Iterable
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from functools import update_wrapper
from functools import wraps
from importlib import import_module
from inspect import signature
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify
from simplify.core.utilities import subsetify


@dataclass
class SimpleEditor(ABC):
    """Base class for creating and applying SimpleManuscript subclasses.

    Args:
        project ('Project'): a related Project instance.
        worker ('Worker'): the Worker instance for which a subclass should edit
            or apply a Book instance.

    """
    project: 'Project'
    worker: 'Worker'

    def __post_init__(self) -> None:
        """Adds attributes from an 'idea' in 'project'."""
        try:
            self = self.project.idea.apply(instance = self)
        except AttributeError:
            pass
        return self

    """ Core siMpLify Methods """

    def draft(self, step: str) -> NotImplementedError:
        """Creates skeleton of a Book instance.

        Args:
            step (str): name of 'step' in Project.

        """
        raise NotImplementedError(' '.join(
            [self.__name__, 'has no draft method. Use Author instead.']))

    def publish(self, step: str) -> NotImplementedError:
        """Finalizes a Book instance and its Chapters and Techniques.

        Args:
            step (str): name of 'step' in Project.

        """
        raise NotImplementedError(' '.join(
            [self.__name__, 'has no publish method. Use Publisher instead.']))

    def apply(self, step: str, data: object) -> NotImplementedError:
        """Applies Book instance to 'data'.

        Args:
            step (str): name of 'step' in Project.
            data (object): data object for a Book instance methods to be
                applied.

        """
        raise NotImplementedError(' '.join(
            [self.__name__, 'has no apply method. Use Scholar instead.']))


@dataclass
class SimpleOutline(Container, ABC):
    """Object construction techniques used by SimpleEditor subclasses.

    Ideally, this class should have no additional methods beyond the lazy
    loader ('load' method) and __contains__ dunder method.

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

    """
    name: str
    module: str

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
        return getattr(import_module(self.module), component)


@dataclass
class SimpleManuscript(Iterable):
    """Base class for Book, Chapter, and Technique iterables.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        iterable (Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter__). Defaults to None.

    """

    name: Optional[str] = None
    iterable: Optional[str] = None

    def __post_init__(self) -> None:
        """Sets 'name' and 'iterable' if not passed.

        Raises:
            ValueError: if 'iterable' is not provided and no default attribute
                is found to store class instance iterables.

        """
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        if self.iterable is None:
            if hasattr(self, 'chapters'):
                self.iterable = 'chapters'
            elif hasattr(self, 'techniques'):
                self.iterable = 'techniques'
            else:
                raise ValueError(' '.join(
                    ['Iterable attribute not found in', self.name]))
        return self

    """ Required ABC Methods """

    def __iter__(self) -> Iterable:
        """Returns class instance iterable."""
        return iter(getattr(self, self.iterable))

    """ Public Methods """

    def add(self, attribute: str, options: Any) -> None:
        """Generic 'add' method' for SimpleManuscripts.

        Users can use the specific instance methods such as 'add_techniques'.
        This method is provided in case a user wants to use a single 'add'
        method with the 'attribute' argument indicating the specific method to
        be called. This might be helpful in certain iteration scenarios.

        Args:
            attribute (str): name of type of object to add to a SimpleManuscript
                instance. This should correspond to a method named:
                'add_[attribute]' in the SimpleManuscript instance.
            options (Any): item(s) to add to a SimpleManuscript instance.

        """
        getattr(self, '_'.join(['add', attribute]))(options)
        return self


@dataclass
class SimpleCatalog(MutableMapping):
    """A flexible dictionary that includes wildcard keys.

    The base class includes 'default', 'all', and 'none' wilcard properties
    which can be accessed through dict methods by those names. Users can also
    set the 'default' and 'none' properties to change what is returned when the
    corresponding keys are sought.

    Args:
        dictionary (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        wildcards (Optional[List[str]]): a list of corresponding properties
            which access sets of dictionary keys. If none is passed, the two
            included properties ('default' and 'all') are used.
        defaults (Optional[List[str]]): a list of keys in 'dictionary' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        null_value (Optional[Any]): value to return when 'none' is accessed or
            an item isn't found in 'dictionary'. Defaults to None.

    """
    dictionary: Optional[Dict[str, Any]] = field(default_factory = dict)
    wildcards: Optional[List[str]] = field(default_factory = list)
    defaults: Optional[List[str]] = field(default_factory = list)
    null_value: Optional[Any] = None

    def __post_init__(self) -> None:
        """Initializes 'defaults' and 'wildcards'."""
        if not self.wildcards:
            self.wildcards = ['all', 'default', 'none']
        if not self.defaults:
            self.defaults = list(self.dictionary.keys())
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns value for 'key' in 'dictionary'.

        If there are no matches, the method searches for a matching wildcard in
        attributes.

        Args:
            key (str): name of key in 'dictionary'.

        Returns:
            Any: item stored as a 'dictionary', a 'wildcard', or 'null_value'.

        """
        try:
            return self.dictionary[key]
        except KeyError:
            if key in self.wildcards:
                return getattr(self, key)
            else:
                return self.null_value

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'dictionary'.

        Args:
            key (str): name of key in 'dictionary'.

        """
        try:
            del self.dictionary[key]
        except KeyError:
            pass
        return self

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'dictionary' to 'value'.

        Args:
            key (str): name of key in 'dictionary'.
            value (Any): value to be paired with 'key' in 'dictionary'.

        """
        self.dictionary[key] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'dictionary'.

        Returns:
            Iterable stored in 'dictionary'.

        """
        return iter(self.dictionary.items())

    def __len__(self) -> int:
        """Returns length of 'dictionary'.

        Returns:
            Integer: length of 'dictionary'.

        """
        return len(self.dictionary)

    """ Wildcard Properties """

    @property
    def all(self) -> Dict[str, Any]:
        """Returns 'dictionary' values.

        Returns:
            List[str] of values stored in 'dictionary'.

        """
        return list(self.dictionary.values())

    @property
    def default(self) -> Dict[str, Any]:
        """Returns key/values for keys in '_default'.

        Returns:
            List[str]: keys stored in 'defaults' of 'dictionary'.

        """
        return subsetify(self.dictionary, self._default)

    @default.setter
    def default(self, keys: Union[List[str], str]) -> None:
        """Sets '_default' to 'dictionary'

        Args:
            keys (Union[List[str], str]): list of keys in 'dictionary' to return
                when 'default' is accessed.

        """
        self._default = listify(keys)
        return self

    @default.deleter
    def default(self, keys: Union[List[str], str]) -> None:
        """Removes 'dictionary' from '_default'.

        Args:
            keys (Union[List[str], str]): list of keys in 'dictionary' to remove
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
class SimpleProgression(SimpleCatalog):
    """A flexible dictionary that keeps wildcards and a separate ordered list.

    Args:
        dictionary (Optional[str, Any]): default stored dictionary. Defaults to
            an empty dictionary.
        order (Optional[List[str]]): the order the keys in 'dictionary' should
            be accessed. Even though python (3.7+) are now ordered, the order
            is dependent upon when an item is added. This attribute allows
            the dictionary to be iterated based upon a separate variable which
            can be updated with the 'order' property. If none is passed, the
            initial order of the keys in 'dictionary' is used.
        wildcards (Optional[List[str]]): a list of corresponding properties
            which access sets of dictionary keys. If none is passed, the two
            included properties ('default' and 'all') are used.
        defaults (Optional[List[str]]): a list of keys in 'dictionary' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        null_value (Optional[Any]): value to return when 'none' is accessed or
            an item isn't found in 'dictionary'. Defaults to None.

    """
    dictionary: Optional[Dict[str, Any]] = field(default_factory = dict)
    wildcards: Optional[List[str]] = field(default_factory = list)
    defaults: Optional[List[str]] = field(default_factory = list)
    null_value: Optional[Any] = None
    order: Optional[List[str]] = field(default_factory = list)

    def __post_init__(self) -> None:
        """Initializes '_order', 'defaults', and 'wildcards'."""
        if self.dictionary and not self.order:
            self._order = list(self.dictionary.keys())
        super().__post_init__()
        return self

    """ Required ABC Methods """

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'dictionary'.

        Args:
            key (str): name of key in 'dictionary'.

        """
        try:
            del self.dictionary[key]
            self.order.remove[key]
        except KeyError:
            pass
        return self

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'dictionary' to 'value'.

        Args:
            key (str): name of key in 'dictionary'.
            value (Any): value to be paired with 'key' in 'dictionary'.

        """
        self.dictionary[key] = value
        self.order.append(key)
        return self

    """ Other Dunder Methods """

    def __add__(self, other: Union['SimpleCatalog', Dict[str, Any]]) -> None:
        """Combines argument with 'dictionary'.

        Args:
            other (Union['SimpleCatalog', Dict[str, Any]]): another
                'SimpleCatalog' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    def __iadd__(self, other: Union['SimpleCatalog', Dict[str, Any]]) -> None:
        """Combines argument with 'dictionary'.

        Args:
            other (Union['SimpleCatalog', Dict[str, Any]]): another
                'SimpleCatalog' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    """ Public Methods """

    def add(self,
            key: Optional[str] = None,
            value: Optional[Any] = None,
            options: Optional[Union[
                'SimpleCatalog', Dict[str, Any]]] = None) -> None:
        """Combines arguments with 'dictionary'.

        Args:
            key (Optional[str]): options key for 'value' to use. Defaults to
                None.
            value (Optional[Any]): item to store in 'options'. Defaults to None.
            options (Optional[Union['SimpleCatalog', Dict[str, Any]]]):
                another 'SimpleCatalog' instance/subclass or a compatible
                dictionary. Defaults to None.

        """
        if key is not None and value is not None:
            self.dictionary[key] = value
            self.order.append(key)
        if options is not None:
            self.update(options = options)
        return self

    """ Dictionary Compatibility Methods """

    def update(self,
            options: Union['SimpleCatalog', Dict[str, Any]] = None) -> None:
        """Combines argument with 'dictionary'.

        Args:
            options ([Union['SimpleCatalog', Dict[str, Any]]): another
                'SimpleCatalog' instance/subclass or a compatible dictionary.
                Defaults to None.

        """
        try:
            self.dictionary.update(getattr(options, options.dictionary))
            self.order.extend(getattr(options, options.order))
        except AttributeError:
            try:
                self.dictionary.update(options)
                self.order.extend(list(options.keys()))
            except (TypeError, AttributeError):
                pass
        return self

    """ Order Property """

    @property
    def order(self) -> List[str]:
        """Returns '_order' or list of keys of 'dictionary'.

        Returns:
            List[str]: keys stored in '_order' of 'dictionary'.

        """
        try:
            self._order = deduplicate(
                [x for x in self._order if x in self.dictionary.keys()])
            return self._order
        except AttributeError:
            self._order = list(self.dictionary.keys())
            return self._order

    @order.setter
    def order(self, keys: Union[List[str], str]) -> None:
        """Sets '_order' to 'dictionary'

        Args:
            keys (Union[List[str], str]): list of keys in 'dictionary' to return
                when '_order' is accessed.

        """
        self._order = listify(keys)
        return self

    @order.deleter
    def order(self, keys: Union[List[str], str]) -> None:
        """Removes 'dictionary' from '_order'.

        Args:
            keys (Union[List[str], str]): list of keys in 'dictionary' to remove
                from '_order'.

        """
        for item in listify(keys):
            try:
                self._order.remove(item)
            except KeyError:
                pass
            except AttributeError:
                self._order = list(self.dictionary.keys())
                self._order.remove(item)
        return self


class SimpleValidator(ABC):
    """Base class decorator to convert arguments to proper types."""

    def __init__(self,
            callable: Callable,
            validators: Optional[Dict[str, Callable]] = None) -> None:
        """Sets initial validator options.

        Args:
            callable (Callable): wrapped method, function, or callable class.
            validators Optional[Dict[str, Callable]]: keys are names of
                parameters and values are functions to convert or validate
                passed arguments. Those functions must return a completed
                object and take only a single passed passed argument. Defaults
                to None.

        """
        self.callable = callable
        update_wrapper(self, self.callable)
        if self.validators is None:
            self.validators = {}
        return self

    """ Required Wrapper Method """

    def __call__(self) -> Callable:
        """Converts arguments of 'callable' to appropriate type.

        Returns:
            Callable: with all arguments converted to appropriate types.

        """
        call_signature = signature(self.callable)
        @wraps(self.callable)
        def wrapper(self, *args, **kwargs):
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            arguments = self.apply(arguments = arguments)
            return self.callable(self, **arguments)
        return wrapper

    """ Core siMpLify Methods """

    def apply(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Converts values of 'arguments' to proper types.

        Args:
            arguments (Dict[str, Any]): arguments with values to be converted.

        Returns:
            Dict[str, Any]: arguments with converted values.

        """
        for argument, validator in self.validators.items():
            try:
                arguments[argument] = validator(arguments[argument])
            except KeyError:
                pass
        return arguments


@dataclass
class SimpleType(MutableMapping, ABC):
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