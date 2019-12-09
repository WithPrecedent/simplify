"""
.. module:: author
:synopsis: composite tree abstract base classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.creator.options import SimpleOptions
from simplify.library.utilities import listify


@dataclass
class Author(object):
    """Base class for building SimpleCodex instances.

    SimpleComposite implements a modified composite tree pattern for organizing
    the various subpackages in siMpLify.

    Args:
        _options (Optional[Union['SimpleOptions', Dict[str, Any]]]): allows
            setting of 'options' property with an argument. Defaults to None.

    """
    _options: (Optional[Union['SimpleOptions', Dict[str, Any]]]) = None

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

@dataclass
class SimpleCodex(ABC):
    """Base class for data processing, analysis, and visualization.

    SimpleComposite implements a modified composite tree pattern for organizing
    the various subpackages in siMpLify.

    Args:
        _options (Optional[Union['SimpleOptions', Dict[str, Any]]]): allows
            setting of 'options' property with an argument. Defaults to None.

    """
    _options: (Optional[Union['SimpleOptions', Dict[str, Any]]]) = None

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns '_children' dictionary as iterable."""
        return iter(self._children)

    """ Private Methods """

    def _draft_options(self) -> None:
        """Subclasses should provide their own methods to create 'options'.

        If the subclasses also allow for passing of '_options', the code below
        should be included as well.Any

        """
        if self._options is None:
            self._options = SimpleOptions(options = {}, _author = self)
        elif isinstance(self._options, Dict):
            self._options = SimpleOptions(
                options = self._options,
                _author = self)
        return self

    def _draft_techniques(self) -> None:
        """If 'techniques' does not exist, gets 'techniques' from 'idea'.

        If there are no matching 'steps' or 'techniques' in 'idea', a list with
        'none' is created for 'techniques'.

        """
        self.compare = False
        if self.techniques is None:
            try:
                self.techniques = getattr(
                    self.idea, '_'.join([self.name, 'steps']))
            except AttributeError:
                try:
                    self.compare = True
                    self.techniques = getattr(
                        self.idea, '_'.join([self.name, 'techniques']))
                except AttributeError:
                    self.techniques = ['none']
        else:
            self.techniques = listify(self.techniques)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Required method that sets default values."""
        # Injects attributes from Idea instance, if values exist.
        self = self.idea.apply(instance = self)
        # initializes core attributes.
        self._draft_options()
        self._draft_techniques()
        return self

    def publish(self, data: Optional[object] = None) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' method.

        Args:
            data (Optional[object]): an optional object needed for the method.

        """
        if data is None:
            data = self.ingredients
        self.options.publish(
            techniques = self.techniques,
            data = data)
        return self

    def apply(self, data: Optional[object], **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (object): data object for methods to be applied.

        """
        for technique in self.techniques:
            data = self.options[technique].options.apply(
                key = technique,
                data = data,
                **kwargs)
        return data

    """ Composite Methods and Properties """

    def add_children(self, keys: Union[List[str], str]) -> None:
        """Adds outline(s) to '_children' from 'options' based on key(s).
        Args:
            keys (Union[List[str], str]): key(s) to 'options'.
        """
        for key in listify(keys):
            self._children[key] = self.options[key]
        return self

    def proxify(self) -> None:
        """Creates proxy names for attributes and methods."""
        try:
            proxy_attributes = {}
            for name, proxy in self.proxies.items():
                for key, value in self.__dict__.items():
                    if name in key:
                        proxy_attributes[key.replace(name, proxy)] = value
            self.__dict__.update(proxy_attributes)
        except AttributeError:
            pass
        return self

    @property
    def parent(self) -> 'SimpleSimpleCodex':
        """Returns '_parent' attribute."""
        return self._parent

    @parent.setter
    def parent(self, parent: 'SimpleSimpleCodex') -> None:
        """Sets '_parent' attribute to 'parent' argument.
        Args:
            parent (SimpleSimpleCodex): SimpleSimpleCodex class up one level in
                the composite tree.
        """
        self._parent = parent
        return self

    @parent.deleter
    def parent(self) -> None:
        """Sets 'parent' to None."""
        self._parent = None
        return self

    @property
    def children(self) -> Dict[str, Union['Outline', 'SimpleSimpleCodex']]:
        """Returns '_children' attribute.
        Returns:
            Dict of str access keys and Outline or SimpleSimpleCodex values.
        """
        return self._children

    @children.setter
    def children(self, children: Dict[str, 'Outline']) -> None:
        """Assigns 'children' to '_children' attribute.
        If 'override' is False, 'children' are added to '_children'.
        Args:
            children (Dict[str, 'Outline']): dictionary with str for reference
                keys and values of 'SimpleSimpleCodex'.
        """
        self._children = children
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
            except KeyError:
                pass
        return self

    """ Strategy Methods and Properties """

    def add_options(self,
            options: Union['SimpleOptions', Dict[str, Any]]) -> None:
        """Assigns 'options' to '_options' attribute.

        Args:
            options (options: Union['SimpleOptions', Dict[str, Any]]): either
                another 'SimpleOptions' instance or an options dict.

        """
        self.options += options
        return self

    @property
    def options(self) -> 'SimpleOptions':
        """Returns '_options' attribute."""
        return self._options

    @options.setter
    def options(self, options: Union['SimpleOptions', Dict[str, Any]]) -> None:
        """Assigns 'options' to '_options' attribute.

        Args:
            options (Union['SimpleOptions', Dict[str, Any]]): SimpleOptions
                instance or a dictionary to be stored within a SimpleOptions
                instance (this should follow the form outlined in the
                SimpleOptions documentation).

        """
        if isinstance(options, dict):
            self._options = SimpleOptions(options = options)
        else:
            self._options.add(options = options)
        return self

    @options.deleter
    def options(self, options: Union[List[str], str]) -> None:
        """ Removes 'options' for '_options' attribute.

        Args:
            options (Union[List[str], str]): key(s) to options classes to
                remove from '_options'.

        """
        for option in listify(options):
            try:
                del self._options[option]
            except KeyError:
                pass
        return self