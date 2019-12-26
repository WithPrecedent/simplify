"""
.. module:: library
:synopsis: strategy container
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from collections.abc import MutableMapping
from dataclasses import dataclass
from importlib import import_module
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


@dataclass
class SimpleCatalog(MutableMapping):
    """Base class for composite tree objects.

    Args:
        project (Optional['Project']): associated Project instance. Defaults to
            None.
        null_value (Any): value to return when 'none' is accessed.

    """
    project: Optional['Project'] = None
    null_value: Optional[Any] = None
    options: Optional[str, Any] = field(default_factory = dict)

    def __post_init__(self) -> None:
        """Sets name of internal 'mapping' dictionary."""
        if not hasattr(self, 'mapping'):
            self.mapping = 'options'
        self.wildcards = ['default', 'all', 'none']
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in the 'mapping' dictionary.

        Args:
            item (str): name of key in the 'mapping' dictionary.

        """
        try:
            del getattr(self, self.mapping)[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> Any:
        """Returns item in the 'mapping' dictionary.

        If there are no matches, the method searches for a matching wildcard in
        attributes.

        Args:
            item (str): name of key in the 'mapping' dictionary.

        Returns:
            Any: item stored as a the 'mapping' dictionary value.

        Raises:
            KeyError: if 'item' is not found in the 'mapping' dictionary.

        """
        try:
            return getattr(self, self.mapping)[item]
        except KeyError:
            if item in self.wildcards:
                return getattr(self, item)
            else:
                raise KeyError(' '.join(
                    [item, 'is not in', self.__class__,__.name__]))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets 'item' in the 'mapping' dictionary to 'value'.

        Args:
            item (str): name of key in the 'mapping' dictionary.
            value (Any): value to be paired with 'item' in the 'mapping'
                dictionary.

        """
        getattr(self, self.mapping)[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of the 'mapping' dictionary.

        Returns:
            Iterable stored in the 'mapping' dictionary.

        """
        return iter(getattr(self, self.mapping))

    def __len__(self) -> int:
        """Returns length of the 'mapping' dictionary if 'iterable' not set..

        Returns:
            Integer of length of 'mapping' dictionary.

        """
        return len(getattr(self, self.mapping))

    """ Other Dunder Methods """

    def __add__(self, other: Union['SimpleComposite', Dict[str, Any]]) -> None:
        """Combines argument with the 'mapping' dictionary.

        Args:
            other (Union['SimpleComposite', Dict[str, Any]]): another
                'SimpleComposite' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    def __iadd__(self, other: Union['SimpleComposite', Dict[str, Any]]) -> None:
        """Combines argument with the 'mapping' dictionary.

        Args:
            other (Union['SimpleComposite', Dict[str, Any]]): another
                'SimpleComposite' instance or compatible dictionary.

        """
        self.add(options = other)
        return self

    """ Public Methods """

    def add(self,
            key: Optional[str] = None,
            value: Optional[Any] = None,
            options: Optional[Union[
                'SimpleComposite', Dict[str, Any]]] = None) -> None:
        """Combines arguments with the 'mapping' dictionary.

        Args:
            key (Optional[str]): dictionary key for 'value' to use. Defaults to
                None.
            value (Optional[Any]): item to store in the 'mapping' dictionary.
                Defaults to None.
            options (Optional[Union['SimpleComposite', Dict[str, Any]]]):
                another 'SimpleComposite' instance or a compatible dictionary.
                Defaults to None.

        """
        if key is not None and value is not None:
            getattr(self, self.mapping)[key] = value
        if options is not None:
            try:
                getattr(self, self.mapping).update(
                    getattr(options, options.mapping))
            except AttributeError:
                try:
                    getattr(self, self.mapping).update(options)
                except (TypeError, AttributeError):
                    pass
        return self

    """ Wildcard Properties """

    @property
    def all(self) -> List[str]:
        """Returns list of keys of the 'mapping' dictionary.

        Returns:
            List[str] of keys stored in the 'mapping' dictionary.

        """
        return list(self.keys())

    @property
    def default(self) -> None:
        """Returns '_default' or list of keys of the 'mapping' dictionary.

        Returns:
            List[str] of keys stored in '_default' or the 'mapping' dictionary.

        """
        try:
            return self._default
        except AttributeError:
            self._default = self.all
            return self._default

    @default.setter
    def default(self, options: Union[List[str], str]) -> None:
        """Sets '_default' to 'options'

        Args:
            'options' (Union[List[str], str]): list of keys in the mapping
                dictionary to return when 'default' is accessed.

        """
        self._default = listify(options)
        return self

    @default.deleter
    def default(self, options: Union[List[str], str]) -> None:
        """Removes 'options' from '_default'.

        Args:
            'options' (Union[List[str], str]): list of keys in the mapping
                dictionary to remove from '_default'.

        """
        for option in listify(options):
            try:
                del self._default[option]
            except KeyError:
                pass
            except AttributeError:
                self._default = self.all
                del self.default[options]
        return self

    @property
    def none(self) -> None:
        """Returns 'null_value'.

        Returns:
            'null_value' attribute or None.

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
class Library(SimpleCatalog):
    """Stores Resource instances.

    Args:
        collection (Optional[Dict[str, 'Resource']]): a dictionary with
            strings as keys and Resource to create Collection instances as
            values. Defaults to an empty dictionary.

    """
    collection: Optional[Dict[str, 'Resource']] = field(default_factory = dict)

    def __post_init__(self) -> None:
        """Sets name of internal 'mapping' dictionary."""
        self.mapping = 'collection'
        return self

    """ Public Methods """

    def draft(self, items: Union[List[str], Dict[str, Any], str]) -> None:
        """Converts selected values in 'mapping' dictionary to Classes.

        Args:
            items (Union[List[str], Dict[str, Any], str]): list of keys,
                dictionary, or a string indicating which 'items' should be
                loaded. If a dictionary is passed, its keys will be used to
                find matches in the 'mapping' dictionary.

        """
        if isinstance(items, dict):
            items = list(items.items())
        for item in listify(items):
            try:
                # Lazily loads all selected Resource instances.
                getattr(self, self.mapping)[item] = getattr(
                    self, self.mapping)[item].load()
            except (KeyError, AttributeError):
                pass
        return self

    def publish(self, items: Union[List[str], Dict[str, Any], str]) -> None:
        """Loads, creates, and finalizes instances in the active dictionary.

        Args:
            items (Union[List[str], Dict[str, Any], str]): list of keys,
                dictionary, or a string indicating which 'items' should be
                instanced. If a dictionary is passed, its keys will be used to
                find matches in the 'mapping' dictionary.

        """
        if isinstance(items, dict):
            items = list(items.items())
        for item in listify(items):
            try:
                instance = getattr(self, self.mapping)[item](
                    project = self.project)
                instance.publish()
                instance = getattr(self, self.mapping)[item] = instance
            except (KeyError, AttributeError):
                pass
        return self


@dataclass
class Resource(Container):
    """Object construction instructions used by SimpleComposite.

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
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify module).
        component (str): name of python object within 'module' to load (can
            either be a siMpLify or non-siMpLify object).

    """
    name: str
    module: str
    component: str

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

    def load(self) -> object:
        """Returns object from module based upon instance attributes.

        Returns:
            object from module indicated in passed Option instance.

        """
        return getattr(import_module(self.module), self.component)


