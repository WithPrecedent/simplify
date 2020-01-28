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
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        iterable (Optional[str]): the name of the attribute that should be
            iterated when a class instance is iterated. Defaults to 'contents'.
        project ('Project'): a related 'Project' instance.

    """
    contents: Optional[Dict[str, Any]] = field(default_factory = dict)
    defaults: Optional[List[str]] = field(default_factory = list)
    iterable: Optional[str] = field(default_factory = lambda: 'contents')
    idea: 'Idea' = None

    def __post_init__(self) -> None:
        """Initializes attributes and settings."""
        # Sets 'defaults' to all keys of 'contents', if not passed.
        self.defaults = self.defaults or list(self.contents.keys())
        # Allows subclasses to customize 'contents' with '_create_contents'.
        self._create_contents()
        # Converts 'contents' to a 'Repository' instance.
        self._nestify()
        # # Creates a pr
        # self._create_proxy_property()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Any:
        """Returns value for 'key' in 'contents'.

        If there are no matches, the method searches for a matching wildcard
        option.

        Args:
            key (str): name of key in 'contents'.

        Returns:
            Any: item stored in 'contents' or a wildcard value.

        """
        try:
            return self.contents[key]
        except KeyError:
            if key in ['all']:
                return list(self.contents.values())
            elif key in ['default', 'defaults']:
                return list(subsetify(self.contents, self.defaults).values())
            elif key in ['none', 'None', 'NONE']:
                return []
            else:
                raise KeyError(' '.join(
                    [key, 'is not in', self.__class__.__name__]))

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (str): name of key in 'contents'.
            value (Any): value to be paired with 'key' in 'contents'.

        """
        self.contents[key] = value
        return self

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

    def __iter__(self) -> Iterable:
        """Returns iterable of attribute named in 'iterable'.

        Returns:
            Iterable stored in attribute named in 'iterable'.

        """
        return iter(getattr(self, self.iterable))

    def __len__(self) -> int:
        """Returns length of attribute named in 'iterable'.

        Returns:
            Integer: length of attribute named in 'iterable'.

        """
        return len(getattr(self, self.iterable))

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

    """ Proxy Property Methods """

    # def _create_proxy_property(self) -> None:
    #     """Creates property named by value of 'iterable' attribute."""
    #     if self.iterable != 'contents':
    #         setattr(self, self.iterable, property(
    #             fget = self._proxy_getter,
    #             fset = self._proxy_setter,
    #             fdel = self._proxy_deleter))
    #     return self

    # def _proxy_getter(self) -> 'Repository':
    #     """Proxy getter for 'contents' using an alias in 'iterable'.

    #     Returns:
    #         'Repository': 'contents' Repository instance.

    #     """
    #     return self.contents

    # def _proxy_setter(self, value: Union['Repository', Dict[str, Any]]) -> None:
    #     """Proxy setter for 'contents' using an alias in 'iterable'.

    #     Args:
    #         value (Union['Repository', Dict[str, Any]]): new mutable mapping
    #             to replace 'contents'.

    #     """
    #     self.contents = value
    #     self._nestify()
    #     return self

    # def _proxy_deleter(self) -> None:
    #     """Proxy deleter using an alias name stored in 'iterable'."""
    #     self.contents = {}
    #     return self

    """ Private Methods """

    def _create_contents(self) -> None:
        """Subclasses should provide their own methods to edit 'contents'."""
        return self

    def _nestify(self) -> None:
        """Converts 1 level of nested dictionaries to Repository instances."""
        for key, value in self.contents.items():
            if isinstance(value, dict):
                self.contents[key] = Repository(contents = value)
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
        if contents is not None:
            self.update(contents)
        self._nestify()
        return self


@dataclass
class Plan(Repository):
    """A wrapper around a Repository that allows for ordered iteration.

    A 'Plan' instance provides an iterable that accesses 'contents' and returns
    the appropriate values in the order of 'steps'.

    Args:
        steps (Optional[List[str]]): an ordred set of steps. Defaults to an
            empty list. All items in 'steps' should correspond to keys in
            'contents' before iterating.
        contents (Optional[Union['Repository', Dict[str, Any]]]): a 'Repository'
            instance or a dictionary that can be used to create one. Defaults to
            an empty Repository.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        iterable (Optional[str]): the name of the attribute that should be
            iterated when a class instance is iterated. Defaults to 'contents'.
        project ('Project'): a related 'Project' instance.
        
    """
    steps: Optional[List[str]] = field(default_factory = list)
    contents: Optional[Union['Repository', Dict[str, Any]]] = field(
        default_factory = Repository)
    defaults: Optional[List[str]] = field(default_factory = list)
    iterable: Optional[str] = field(default_factory = lambda: 'steps')
    idea: 'Idea' = None

    def __post_init__(self) -> None:
        """Initializes attributes if not passed."""
        # Allows subclasses to customize 'contents' with '_create_contents'.
        self._create_contents()
        # Sets 'steps' to all keys in 'contents' if not passed.
        self.steps = self.steps or list(self.contents.keys())
        # Sets 'defaults' to 'steps' if not passed.
        self.defaults = self.defaults or self.steps
        # Converts 'contents' to a 'Repository' instance.
        self._nestify()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: Union[str, int]) -> Any:
        """Returns 'steps' or a wildcard option.

        Args:
            key (Union[str, int]): item, wilcard, or index of steps stored in
                'steps'.

        Returns:
            Any: a whole or part of a Repository values with key(s) matching
                'key'.

        """
        if key in self.steps:
            return self.contents[key]
        if key in ['all']:
            return list(subsetify(self.repository, self.steps).values())
        elif key in ['default', 'defaults']:
            temp = subsetify(self.contents, self.defaults)
            return list(subsetify(temp, self.steps).values())
        elif key in ['none', 'None', 'NONE']:
            return []
        else:
            raise KeyError(' '.join(
                [key, 'is not in', self.__class__.__name__]))

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (str): name of key in 'contents'.
            value (Any): value to be paired with 'key' in 'contents'.

        """
        if key in ['default', 'defaults']:
            self.defaults = value
        elif not key in self.contents:
            self.steps.append(key)
            self.contents[key] = value
        else:
            self.contents[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'contents'.

        Args:
            key (str): name of key in 'contents'.

        """
        try:
            del self.contents[key]
            self.steps.remove(key)
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable from 'contents'.

        Returns:
            Iterable: the portion of 'contents' with keys matching 'steps' in
                the order of 'steps'.

        """
        return iter(subsetify(self.contents, self.steps))

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
            self.add_steps(steps = key)
        if contents is not None:
            self.update(contents)
            self._add_steps(list(contents.keys()))
        self._nestify()
        return self

    def add_steps(self, steps: Union[List[str], str]) -> None:
        """Combines 'steps' with 'steps' attribute.

        Args:
            steps (Union[List[str], str]): step(s) to add to the end of
                the 'steps' attribute.

        """
        if isinstance(steps, str):
            self.steps.append(steps)
        else:
            self.steps.extend(steps)
        return self

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
            self.contents[step] = value
        return self


""" Creation Functions """

# def create_plan(
#         plan: Union['Plan', List[str], str],
#         repository: 'Repository',
#         **kwargs) -> 'Plan':
#     if isinstance(plan, Plan):
#         return plan
#     elif isinstance(plan, (list, str)):
#         return Plan(
#             steps = listify(plan, default_empty = True),
#             contents = repository,
#             **kwargs)
#     else:
#         raise TypeError('plan must be Plan, list, or str type')

# def create_repository(
#         repository: Union['Repository', Dict[str, Any]],
#         **kwargs) -> 'Repository':
#     if isinstance(repository, Repository):
#         return repository
#     elif isinstance(repository, dict):
#         return Repository(contents = repository, **kwargs)
#     else:
#         raise TypeError('repository must be Repository or dict type')