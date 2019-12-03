"""
.. module:: composite
:synopsis: base class for composite tree objects
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.base.options import SimpleOptions
from simplify.core.utilities import listify


@dataclass
class SimpleManuscript(SimpleOptions, ABC):
    """Base class for data processing, analysis, and visualization.

    SimpleComposite implements a modified composite tree pattern for organizing
    the various subpackages in siMpLify.

    """
    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Calls SimpleOptions __post_init__ method.
        SimpleOptions.__post_init__()
        # Injects attributes from Idea instance, if values exist.
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        # Creates proxy names for attributes, if 'proxies' attribute exists.
        self.proxify()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __iter__(self) -> NotImplementedError:
        raise NotImplementedError(' '.join([
            self.__class__.__name__, 'has no child classes']))

    """ Composite Management Methods """

    def proxify(self, proxies: Optional[Dict[str, str]] = None) -> None:
        """Creates proxy names for attributes and methods.

        Args:
            proxies (Optional[Dict[str, str]]): dictionary with keys of current
                attribute names and values of proxy attribute names. If
                'proxies' is not passed, the method looks for 'proxies'
                attribute in the subclass instance. Defaults to None

        """
        if proxies is None:
            try:
                proxies = self.proxies
            except AttributeError:
                pass
        if proxies is not None:
            proxy_attributes = {}
            for name, proxy in proxies.items():
                for key, value in self.__dict__.items():
                    if proxy in key:
                        proxy_attributes[key.replace(name, proxy)] = value
            self.__dict__.update(proxy_attributes)
        return self

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        """Required method that sets default values.

        Subclasses should provide their own 'draft' method.

        """
        return self

    @abstractmethod
    def publish(self, data: Optional[object] = None) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' method.

        Args:
            data (Optional[object]): an Ingredients instance.

        """
        return self

    @abstractmethod
    def apply(self, data: object, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (object): data object for methods to be applied.

        """
        return self

    """ Properties """

    @property
    def parent(self) -> 'Book':
        """Returns '_parent' attribute."""
        return self._parent

    @parent.setter
    def parent(self, parent: 'Book') -> None:
        """Sets '_parent' attribute to 'parent' argument.

        Args:
            parent (Book): Book class up one level in the composite tree.

        """
        self._parent = parent
        return self

    @parent.deleter
    def parent(self) -> None:
        """Sets 'parent' to None."""
        self._parent = None
        return self

    @property
    def children(self) -> Dict[str, Union['Book', 'Page']]:
        """Returns '_children' attribute.

        Returns:
            Dict of str access keys and Book or Page values.

        """
        return self._children

    @children.setter
    def children(self,
            children: Dict[str, Union['Book', 'Page']],
            override: Optional[bool] = False) -> None:
        """Adds 'children' to '_children' attribute.

        If 'override' is True, 'children' replaces '_children'.

        Args:
            children (Dict[str, Union['Book', 'Page']]): dictionary with str
                for reference keys and values of either 'Book' or 'Page'.
            override (Optional[bool]): whether to overwrite existing '_children'
                (True) or add 'children' to '_children' (False). Defaults to
                False.

        """
        if override or not hasattr(self, '_children') or not self._children:
            self._children = children
        else:
            self._children.update(children)
        return self

    @children.deleter
    def children(self, children: Union[List[str], str]) -> None:
        """ Removes 'children' for '_children' attribute.

        Args:
            children (Union[List[str], str]): key(s) to children classes to
                remove from '_children'.

        """
        for child in listify(children):
            try:
                del self._children[child]
            except (KeyError, AttributeError):
                pass
        return self