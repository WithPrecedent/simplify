"""
.. module:: repository
:synopsis: siMpLify base mapping classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify
from simplify.core.utilities import subsetify


@dataclass
class Repository(MutableMapping):
    """Dictionary which accepts lists and wildcards as keys, returns lists.

    The base class includes 'default', 'all', and 'none' wilcard properties.

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
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.

    """
    name: Optional[str] = None
    contents: Optional[Dict[str, Any]] = field(default_factory = dict)
    defaults: Optional[List[str]] = field(default_factory = list)
    idea: ClassVar['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes attributes and settings."""
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Allows subclasses to customize 'contents' with 'create'.
        self.create()
        # Stores 1-level nested dict as a Repository instance.
        self.nestify()
        # Declares list of wildcards for external reference.
        self.wildcards = ['all', 'default', 'none']
        # Sets 'default' to all keys of 'contents', if not passed.
        self.defaults = self.defaults or list(self.contents.keys())
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: Union[List[str], str]) -> List[Any]:
        """Returns value for 'key' in 'contents'.

        If there are no matches, the method searches for a matching wildcard
        option.

        Args:
            key (Union[List[str], str]): name(s) of key(s) in 'contents'.

        Returns:
            List[Any]: item(s) stored in 'contents' or a wildcard value.

        """
        if key in ['all', ['all']]:
            return list(self.contents.values())
        elif key in ['default', ['default']]:
            return list(self.subsetify(self.defaults).values())
        elif key in ['none', ['none']]:
            return []
        else:
            try:
                return list(self.subsetify(listify(key)).values())
            except KeyError:
                raise KeyError(' '.join([key, 'is not in', self.name]))

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (str): name of key in 'contents'.
            value (Any): value to be paired with 'key' in 'contents'.

        """
        if key in ['default']:
            self.defaults = value
        else:
            self.contents[key] = value
        return self

    def __delitem__(self, key: Union[List[str], str]) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (Union[List[str], str]): name(s) of key(s) in 'contents'.

        """
        self.contents = {
            i: self.contents[i] for i in self.contents if i not in listify(key)}
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'contents'.

        Returns:
            Iterable: of 'contents'.

        """
        return iter(self.contents)

    def __len__(self) -> int:
        """Returns length of 'contents'.

        Returns:
            Integer: length of 'contents'.

        """
        return len(self.contents)

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

    def __repr__(self) -> str:
        """Returns '__str__' representation.

        Returns:
            str: default dictionary representation of 'contents'.

        """
        return self.__str__()

    def __str__(self) -> str:
        """Returns default dictionary representation of contents.

        Returns:
            str: default dictionary representation of 'contents'.

        """
        return self.contents.__str__()

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
        if contents is not None:
            self.contents.update(contents)
        self.nestify()
        return self

    def create(self) -> None:
        """Subclasses should provide their own methods to create 'contents'."""
        return self

    def flatten(self) -> None:
        """Moves 1-level nested dict to outer level with tuple key."""
        new_contents = {}
        for outer_key, outer_value in self.contents.items():
            if isinstance(value, dict):
                for inner_key, inner_value in outer_value.items():
                    new_contents[(outer_key, inner_key)] = inner_value
            else:
                new_contents[key] = value
        self.contents = new_contents
        return self

    def nestify(self) -> None:
        """Converts 1 level of nesting to Repository instances."""
        for key, value in self.contents.items():
            if isinstance(value, dict):
                self.contents[key] = Repository(contents = value)
        return self

    def subsetify(self, key: Union[List[str], str]) -> 'Repository':
        """Returns a subset of a Repository

        Args:
            key (Union[List[str], str]): key(s) to get key/values.

        Returns:
            'Repository': with only keys in 'key'.

        """
        return Repository(contents = {i: dictionary[i] for i in self.contents})


@dataclass
class Plan(MutableSequence):
    """

    Args:
        steps (Optional[List[str]]): an ordred set of steps. Defaults to an
            empty list. All items in 'steps' should correspond to keys in
            'repository' before iterating.
        repository ('Repository'): instance with options for 'steps'.
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.

    """
    steps: Union[List[str], str]
    repository: 'Repository'
    idea: ClassVar['Idea'] = None

    def __post_init__(self) -> None:
        self.create()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns 'steps' or a wildcard option.

        Args:
            key (str): item, wilcard, or index of steps stored in
                'steps'.

        Returns:
            Any: a whole or part of a Repository values with key(s) matching
                'key'.

        """
        return self.repository[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'repository' to 'value'.

        Args:
            key (str): name of key in 'repository'.
            value (Any): value to be paired with 'key' in 'repository'.

        """
        self.repository[key] = value
        if not key in self.steps:
            self.steps.append(key)
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'repository'.

        Args:
            key (str): name of key in 'repository'.

        """
        try:
            del self.repository[key]
            self.steps.remove(key)
        except (KeyError, ValueError):
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable from 'repository'.

        Returns:
            Iterable: the portion of 'repository' with keys matching 'steps' in
                the order of 'steps'.

        """
        return iter(self.repository.subsetify(self.steps))

    def __len__(self) -> int:
        """Returns length of attribute named in 'iterable'.

        Returns:
            Integer: length of attribute named in 'iterable'.

        """
        return len(self.steps)

    def insert(self,
            index: int,
            step: str,
            value: Optional[Any] = None) -> None:
        """Inserts item in 'steps' at 'index'.

        Args:
            index (int): location in 'steps' to insert 'step'.
            step (str): step to insert at 'index' in 'steps'.

        """
        self.steps.insert(index, step)
        if value is not None:
            self.repository[step] = value
        return self

    """ Other Dunder Methods """

    def __add__(self, other: Union['Repository', Dict[str, Any]]) -> None:
        """Combines argument with 'repository'.

        Args:
            other (Union['Repository', Dict[str, Any]]): another
                'Repository' instance or compatible dictionary.

        """
        self.add(repository = other)
        return self

    def __iadd__(self, other: Union['Repository', Dict[str, Any]]) -> None:
        """Combines argument with 'repository'.

        Args:
            other (Union['Repository', Dict[str, Any]]): another
                'Repository' instance or compatible dictionary.

        """
        self.add(repository = other)
        return self

    def __repr__(self) -> str:
        """Returns '__str__' representation.

        Returns:
            str: default dictionary representation of 'repository'.

        """
        return self.__str__()

    def __str__(self) -> str:
        """Returns default dictionary representation of repository.

        Returns:
            str: default dictionary representation of 'repository'.

        """
        return self.repository.__str__()

    """ Public Methods """

    def add(self, steps: Union[List[str], str]) -> None:
        """Combines 'steps' with 'steps' attribute.

        Args:
            steps (Union[List[str], str]): step(s) to add to the end of the
                'steps' attribute.

        """
        self.steps.extend(listify(steps))
        return self

    def create(self) -> None:
        """Subclasses may provide their own methods to create 'steps'."""
        self.steps = listify(self.steps, default_empty = True)
        if not self.steps:
            try:
                key = self.repository.name
                self.steps = self.idea[key]['_'.join([key, 'steps'])]
            except (KeyError, AttributeError):
                pass
        return self


@dataclass
class Outline(Container):
    """Object construction instructions used by Publisher subclasses.

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
                    [getattr(self, component), 'is neither in', self.module,
                        'nor', self.default_module]))