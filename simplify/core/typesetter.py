"""
.. module:: typesetter
:synopsis: base abstract base classes for siMpLify
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import Container
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify


@dataclass
class SimpleOptions(MutableMapping):
    """Mixin base class for classes with 'options' attribute."""

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


@dataclass
class SimpleBuilder(ABC, SimpleOptions):
    """Base class for building objects."""

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
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        """Creates builder instance default settings.

        Subclass instances should provide their own methods.

        """
        return self

    @abstractmethod
    def publish(self) -> None:
        """Finalizes built object.

        Subclass instances should provide their own methods.

        """
        return self

    @abstractmethod
    def apply(self, data: object, *args, **kwargs) -> object:
        """Returns built object.

        Subclass instances should provide their own methods.

        """
        return data


@dataclass
class SimpleComposite(ABC, SimpleOptions):
    """Base class for data processing classes.

    SimpleComposite implements a modified composite tree pattern for organizing
    Projects, Books, Chapters, and Pages. The hierarchy between each level is
    fixed, but the core methods are shared by all of the levels in that
    hierarchy.

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
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable 'children'."""
        try:
            return iter(self.children)
        except AttributeError:
            self.children = {}
            return iter(self.children)


    """ Composite Management Methods """

    def add_parent(self, parent: 'SimpleComposite') -> None:
        """Sets 'parent' attribute to 'SimpleComposite'.

        Setting 'parent' allows different SimpleComposite instances to access
        attributes from other SimpleComposite instances that are connected
        through parent and/or child relationships.

        Args:
            parent ('SimpleComposite'): SimpleComposite subclass above another
                SimpleComposite subclass instance in the composite tree.

        """
        self.parent = parent
        return self

    def remove_parent(self) -> None:
        """Sets 'parent' to None."""
        self.parent = None
        return self

    def add_child(self, name: str, child: 'SimpleComposite') -> None:
        """Adds 'child' instance to 'children'.

        Args:
            name (str): key name for child instance to be accessed from
                'children' dict.
            child ('SimpleComposite'): SimpleComposite subclass below another
                SimpleComposite subclass instance in the composite tree.

        """
        try:
            self.children[name] = child
        except (AttributeError, TypeError):
            self.children = {}
            self.children[name] = child
        return self

    def remove_child(self, name: str) -> None:
        """Removes a child instance link from 'children'.

        Args:
            name (str): key name of child instance to remove.

        """
        try:
            del self.children[name]
        except KeyError:
            pass
        return self

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        """Required method that sets default values.

        Subclasses should provide their own 'draft' method.

        """
        return self

    @abstractmethod
    def publish(self, data: Optional['Ingredients'] = None) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' method.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.

        """
        return self

    @abstractmethod
    def apply(self, data: 'Ingredients', **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (Ingredients): data object for methods to be applied.

        """
        return self


@dataclass
class SimpleContainer(Container):
    """Base class for simple data containers."""

    """ Required ABC Methods """

    def __contains__(self, attribute: str) -> bool:
        """Returns whether attribute exists in class instance.

        Args:
            attribute (str): name of attribute to check.

        Returns:
            bool: whether the attribute exists.

        """
        return hasattr(self, attribute)


@dataclass
class SimpleFile(ABC, SimpleOptions):
    """Base class for storing and creating file paths."""

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Calls SimpleOptions __post_init__ method.
        SimpleOptions.__post_init__()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Public Methods """

    def load(self,
            name: Optional[str] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
        """Loads object from file into the subclass attribute 'name'.

        For any arguments not passed, default values stored in the shared
        Library instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            name (Optional[str]): name of attribute for the file contents to be
                stored. Defaults to None.
            file_path (Optional[str]): a complete file path for the file to be
                loaded. Defaults to None.
            folder (Optional[str]): a path to the folder where the file should
                be loaded from (not used if file_path is passed). Defaults to
                None.
            file_name (Optional[str]): contains the name of the file to be
                loaded without the file extension (not used if file_path is
                passed). Defaults to None.
            file_format (Optional[str]): name of file format in
                library.extensions. Defaults to None.

        """
        setattr(self, name, self.library.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format))
        return self

    def save(self,
            variable: Optional[Union['SimpleComposite', str]] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
        """Exports a variable or attribute to disk.

        If 'variable' is not passed, 'self' will be used.

        For other arguments not passed, default values stored in the shared
        library instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            variable (Optional[Union['SimpleComposite'], str]): a python object
                or a string corresponding to a subclass attribute which should
                be saved to disk. Defaults to None.
            file_path (Optional[str]): a complete file path for the file to be
                saved. Defaults to None.
            folder (Optional[str]): a path to the folder where the file should
                be saved (not used if file_path is passed). Defaults to None.
            file_name (Optional[str]): contains the name of the file to be saved
                without the file extension (not used if file_path is passed).
                Defaults to None.
            file_format (Optional[str]): name of file format in
                library.extensions. Defaults to None.

        """
        # If variable is not passed, the subclass instance is saved.
        if variable is None:
            variable = self
        # If a string, 'variable' is converted to a local attribute with the
        # string as its name.
        else:
            try:
                variable = getattr(self, variable)
            except TypeError:
                pass
        self.library.save(
            variable = variable,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format)
        return self


@dataclass
class SimpleState(ABC, SimpleOptions):
    """Base class for state management."""

    def _post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Calls SimpleOptions __post_init__ method.
        SimpleOptions.__post_init__()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'state'."""
        return self.publish()

    def __str__(self) -> str:
        """Returns string name of 'state'."""
        return self.publish()

    """ State Management Methods """

    def change(self, new_state: str) -> None:
        """Changes 'state' to 'new_state'.

        Args:
            new_state(str): name of new state matching a string in 'options'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.options:
            self.state = new_state
        else:
            error = ' '.join([new_state, 'is not a recognized state'])
            raise TypeError(error)

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        """Creates state machine default settings.

        Subclass instances should provide their own methods.

        """
        return self

    def publish(self) -> str:
        """Returns current state in 'state' attribute."""
        return self.state
