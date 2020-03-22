"""
.. module:: base
:synopsis: abstract base classes for siMpLify
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import abc
import collections.abc
import dataclasses
import importlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


@dataclasses.dataclass
class SimpleSystem(abc.ABC):
    """Base class for a siMpLify workflow.

    A 'SimpleSystem' subclass maintains a progress state stored in the attribute
    'stage'. The 'stage' corresponds to whether one of the core workflow
    methods has been called. The string stored in 'stage' can then be used by
    subclasses to alter instance behavior, call methods, or change access
    method functionality.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class instance
            needs settings from the shared 'Idea' instance, 'name' should match
            the appropriate section name in that 'Idea' instance. When
            subclassing, it is a good idea to use the same 'name' attribute as
            the base class for effective coordination between siMpLify classes.
            'name' is used instead of __class__.__name__ to make such
            subclassing easier. Defaults to None or __class__.__name__.lower().
        stages (Optional[List[str]]): list of recognized states which correspond
            to methods within a class instance. Defaults to ['initialize',
            'draft', 'publish', 'apply'].

    """
    name: Optional[str] = None
    stages: Optional[List[str]] = dataclasses.field(
        default_factory = lambda: ['initialize', 'draft', 'publish', 'apply'])

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Sets 'name' to default value if it is not passed.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Sets initial stage.
        self.stage = self.stages[0]
        return self

    """ Factory Method """

    @classmethod
    def create(cls, *args, **kwargs) -> 'SimpleSystem':
        """Returns a class object based upon arguments passed.

        This is a placeholder that returns a basic version of the class.
        Subclasses should provide alternate methods for more complicated
        construction.

        """
        return cls(*args, **kwargs)

    """ Required Methods """

    @abc.abstractmethod
    def add(self, item: Union[
        'SimpleSystem', 'SimpleContainer', 'SimpleComponent']) -> None:
        """Subclasses must provide their own methods."""
        return self

    @abc.abstractmethod
    def draft(self) -> None:
        """Subclasses must provide their own methods."""
        return self

    @abc.abstractmethod
    def publish(self) -> None:
        """Subclasses must provide their own methods."""
        return self

    @abc.abstractmethod
    def apply(self, *args, **kwargs) -> None:
        """Subclasses must provide their own methods."""
        return self

    """ Dunder Methods """

    def __getattribute__(self, attribute: str) -> Any:
        """Changes 'stage' if one of the corresponding methods are called.

        If attribute matches any item in 'stages', the 'stage' attribute is
        assigned to 'attribute.'

        Args:
            attribute (str): name of attribute sought.

        """
        try:
            if attribute in super().__getattribute__('stages'):
                super().__getattribute__('advance')(stage = attribute)
        except AttributeError:
            pass
        return super().__getattribute__(attribute)

    """ Stage Management Method """

    def advance(self, stage: Optional[str] = None) -> None:
        """Advances to next stage in 'stages' or to 'stage' argument.

        Args:
            stage(Optional[str]): name of stage matching a string in 'stages'.

        Raises:
            ValueError: if 'stage' is neither None nor in 'stages'.

        """
        self.previous_stage = self.stage
        if stage is None:
            try:
                self.stage = self.stages[self.stages.index(self.stage) + 1]
            except IndexError:
                pass
        elif stage in self.stages:
            self.stage = stage
        else:
            raise ValueError(' '.join([stage, 'is not a recognized stage']))
        return self


@dataclasses.dataclass
class SimpleCreator(abc.ABC):
    """Base class for creating or modifying other siMpLify classes.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class instance
            needs settings from the shared 'Idea' instance, 'name' should match
            the appropriate section name in that 'Idea' instance. When
            subclassing, it is a good idea to use the same 'name' attribute as
            the base class for effective coordination between siMpLify classes.
            'name' is used instead of __class__.__name__ to make such
            subclassing easier. Defaults to None or __class__.__name__.lower().

    """
    name: Optional[str] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Sets 'name' to default value if it is not passed.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        return self

    """ Factory Method """

    @classmethod
    def create(cls, *args, **kwargs) -> 'SimpleCreator':
        """Returns a class object based upon arguments passed.

        This is a placeholder that returns a basic version of the class.
        Subclasses should provide alternate methods for more complicated
        construction.

        """
        return cls(*args, **kwargs)

    """ Required Subclass Methods """

    @abc.abstractmethod
    def apply(self,
            data: Union[
                'SimpleSystem',
                'SimpleContainer',
                'SimpleComponent'],
            **kwargs) -> Union[
                    'SimpleSystem',
                    'SimpleContainer',
                    'SimpleComponent']:
        """Subclasses must provide their own methods."""
        return self


@dataclasses.dataclass
class SimpleRepository(collections.abc.MutableMapping):
    """Base class for policy and option storage.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class instance
            needs settings from the shared 'Idea' instance, 'name' should match
            the appropriate section name in that 'Idea' instance. When
            subclassing, it is a good idea to use the same 'name' attribute as
            the base class for effective coordination between siMpLify classes.
            'name' is used instead of __class__.__name__ to make such
            subclassing easier. Defaults to None or __class__.__name__.lower().
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        wildcards (Optional[List[str]]): a list of wildcard keys which return
            lists of values. Defaults to ['all', 'default', 'none'].
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.

    """
    name: Optional[str] = None
    contents: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory = dict)
    wildcards: Optional[List[str]] = dataclasses.field(
        default_factory = lambda: ['all', 'default', 'none'])
    defaults: Optional[List[str]] = dataclasses.field(default_factory = list)

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Sets 'name' to default value if it is not passed.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Allows subclasses to customize 'contents' with 'create'.
        self = self.create(
            name = self.name,
            contents = self.contents,
            defaults = self.defaults,
            wildcards = self.wildcards)
        # Stores nested dictionaries as 'SimpleRepository' instances.
        self.nestify()
        # Sets 'default' to all keys of 'contents', if not passed.
        self.defaults = self.defaults or list(self.contents.keys())
        return self

    """ Factory Method """

    @classmethod
    def create(cls, *args, **kwargs) -> 'SimpleRepository':
        """Returns a class object based upon arguments passed.

        This is a placeholder that returns a basic version of the class.
        Subclasses should provide alternate methods for more complicated
        construction.

        """
        return cls(*args, **kwargs)

    """ Required ABC Methods """

    def __getitem__(self, key: Union[List[str], str]) -> List[Any]:
        """Returns value(s) for 'key' in 'contents'.

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
            return list(self.utilities.subsetify(keys = self.defaults).values())
        elif key in ['none', ['none']]:
            return []
        else:
            try:
                return self.contents[key]
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
            key (Union[List[str], str]): name(s) of key(s) in 'contents' to
                delete the key/value pair.

        """
        self.contents = {
            i: self.contents[i]
            for i in self.contents if i not in utilities.listify(key)}
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

    def __add__(self,
            other: Union['SimpleRepository', Dict[str, Any]]) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union['SimpleRepository', Dict[str, Any]]): another
                'SimpleRepository' instance or compatible dictionary.

        """
        self.add(contents = other)
        return self

    def __iadd__(self,
            other: Union['SimpleRepository', Dict[str, Any]]) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union['SimpleRepository', Dict[str, Any]]): another
                'SimpleRepository' instance or compatible dictionary.

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

    """ Construction Method """

    def add(self, contents: Union['SimpleRepository', Dict[str, Any]]) -> None:
        """Combines arguments with 'contents'.

        Args:
            contents (Union['SimpleRepository', Dict[str, Any]]): another
                'SimpleRepository' instance/subclass or a compatible dictionary.

        """
        self.contents.update(contents)
        self.contents = self.nestify(contents = self.contents)
        return self

    """ Structural Methods """

    def nestify(self,
            contents: Union[
                'SimpleRepository',
                Dict[str, Any]]) -> 'SimpleRepository':
        """Converts nested dictionaries to 'SimpleRepository' instances.

        Args:
            contents (Union['SimpleRepository', Dict[str, Any]]): mutable
                mapping to be converted to a 'SimpleRepository' instance.

        Returns:
            'SimpleRepository': subclass instance with 'contents' stored.

        """
        new_repository = self.__new__(wildcards = self.wildcards)
        for key, value in contents.items():
            if isinstance(value, dict):
                new_repository.add(
                    contents = {key: self.nestify(contents = value)})
            else:
                new_repository.add(contents = {key: value})
        return new_repository

    def subsetify(self, keys: Union[List[str], str]) -> 'SimpleRepository':
        """Returns a subset of a SimpleRepository.

        Args:
            keys (Union[List[str], str]): key(s) to get key/values.

        Returns:
            'SimpleRepository': with only keys in 'keys'.

        """
        return self.__new__(contents = {
                i: self.contents.get(i) for i in utilities.listify(keys)})


@dataclasses.dataclass
class SimplePlan(collections.abc.MutableSequence):
    """

    Args:
        items (Optional[Union[List[], str]): an ordred set of items. Defaults
            to an empty list.
        idea (Optional['Idea']): shared 'Idea' instance with project settings.

    """
    items: Optional[Union[
        List[Union[
            'SimpleComponent',
            'SimpleCreator',
            str]],
        'SimpleComponent',
        'SimpleCreator',
        str]] = dataclasses.field(default_factory = list)
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self.items = utilities.listify(self.items)
        return self

    """ Factory Method """

    @classmethod
    def create(cls, *args, **kwargs) -> 'SimpleContainer':
        """Returns a class object based upon arguments passed.

        This is a placeholder that returns a basic version of the class.
        Subclasses should provide alternate methods for more complicated
        construction.

        """
        return cls(*args, **kwargs)

    """ Required ABC Methods """

    def __getitem__(self, index: Union[int, str, slice]) -> Any:
        """Returns 'items' or a wildcard option.

        Args:
            key (Union[int, str, slice]):

        Returns:
            Any: a whole or part of a Repository values with key(s) matching
                'key'.

        """
        try:
            return self.items[index]
        except TypeError:
            return self.items.index(index)


    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'repository' to 'value'.

        Args:
            key (str): name of key in 'repository'.
            value (Any): value to be paired with 'key' in 'repository'.

        """
        self.repository[key] = value
        if key not in self.items:
            self.items.append(key)
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'repository'.

        Args:
            key (str): name of key in 'repository'.

        """
        try:
            del self.repository[key]
            self.items.remove(key)
        except (KeyError, ValueError):
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable from 'repository'.

        Returns:
            Iterable: the portion of 'repository' with keys matching 'items' in
                the order of 'items'.

        """
        return iter(self.items)

    def __len__(self) -> int:
        """Returns length of attribute named in 'iterable'.

        Returns:
            Integer: length of attribute named in 'iterable'.

        """
        return len(self.items)

    def insert(self,
            index: int,
            item: str,
            value: Optional[Any] = None) -> None:
        """Inserts item in 'items' at 'index'.

        Args:
            index (int): location in 'items' to insert 'item'.
            item (str): item to insert at 'index' in 'items'.

        """
        self.items.insert(index, item)
        if value is not None:
            self.repository[item] = value
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

    def add(self, items: Union['SimpleContainer', 'SimpleComponent']) -> None:
        """Combines 'items' with 'items' attribute.

        Args:
            items (Union[List[str], str]): item(s) to add to the end of the
                'items' attribute.

        """
        self.items.extend(utilities.listify(items))
        return self


@dataclasses.dataclass
class SimpleComponent(ABC):
    """Base class for lazy loaders for low-level siMpLify objects.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class instance
            needs settings from the shared 'Idea' instance, 'name' should match
            the appropriate section name in that 'Idea' instance. When
            subclassing, it is a good idea to use the same 'name' attribute as
            the base class for effective coordination between siMpLify classes.
            'name' is used instead of __class__.__name__ to make such
            subclassing easier. Defaults to None or __class__.__name__.lower().
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        default_module (Optional[str]): name of a backup module where object to
            use is located (can either be a siMpLify or non-siMpLify module).
            Defaults to 'simplify.core'. Subclasses should not generally
            override this attribute. It allows the 'load' method to use generic
            classes if the specified one is not found.

    """
    name: Optional[str] = None
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.core')
    default_module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.core')

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Sets 'name' to default value if it is not passed.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        return self

    """ Factory Method """

    @classmethod
    def create(cls, *args, **kwargs) -> 'SimpleComponent':
        """Returns a class object based upon arguments passed.

        This is a placeholder that returns a basic version of the class.
        Subclasses should provide alternate methods for more complicated
        construction.

        """
        return cls(*args, **kwargs)

    """ Core siMpLify Methods """

    def load(self, attribute: str) -> object:
        """Returns object named in 'attribute'.

        If 'attribute' is not a str, it is assumed to have already been loaded
        and is returned as is.

        The method searches both 'module' and 'default_module' for the named
        'attribute'.

        Args:
            attribute (str): name of local attribute to load from 'module' or
                'default_module'.

        Returns:
            object: from 'module' pr 'default_module'.

        """
        # If 'attribute' is a string, attempts to load from 'module' or, if not
        # found there, 'default_module'.
        if isinstance(getattr(self, attribute), str):
            try:
                return getattr(
                    importlib.import_module(self.module),
                    getattr(self, attribute))
            except (ImportError, AttributeError):
                try:
                    return getattr(
                        importlib.import_module(self.default_module),
                        getattr(self, attribute))
                except (ImportError, AttributeError):
                    raise ImportError(' '.join(
                        [getattr(self, attribute), 'is neither in',
                            self.module, 'nor', self.default_module]))
        # If 'attribute' is not a string, it is returned as is.
        else:
            return getattr(self, attribute)


@dataclasses.dataclass
class SimpleProxy(abc.ABC):
    """Mixin which creates proxy name for an instance attribute.

    The 'proxify' method dynamically creates a property to access the stored
    attribute. This allows class instances to customize names of stored
    attributes while still using base siMpLify classes.

    """

    """ Proxy Property Methods """

    def _proxy_getter(self) -> Any:
        """Proxy getter for '_attribute'.

        Returns:
            Any: value stored at '_attribute'.

        """
        return getattr(self, self._attribute)

    def _proxy_setter(self, value: Any) -> None:
        """Proxy setter for '_attribute'.

        Args:
            value (Any): value to set attribute to.

        """
        setattr(self, self._attribute, value)
        return self

    def _proxy_deleter(self) -> None:
        """Proxy deleter for '_attribute'."""
        setattr(self, self._attribute, self._default_proxy_value)
        return self

    """ Other Private Methods """

    def _proxify_attribute(self, proxy: str) -> None:
        """Creates proxy property for 'attribute'.

        Args:
            proxy (str): name of proxy property to create.

        """
        setattr(self, proxy, property(
            fget = self._proxy_getter,
            fset = self._proxy_setter,
            fdel = self._proxy_deleter))
        return self

    def _proxify_methods(self, proxy: str) -> None:
        """Creates proxy method with an alternate name.

        Args:
            proxy (str): name of proxy to repalce in method names.

        """
        for item in dir(self):
            if (self.attribute in item
                    and not item.startswith('__')
                    and callabe(item)):
                self.__dict__[item.replace(self.attribute, proxy)] = (
                    getattr(self, item))
        return self

    """ Public Methods """

    def proxify(self,
                proxy: str,
                attribute: str,
                default_value: Optional[Any] = None,
                proxify_methods: Optional[bool] = True) -> None:
        """Adds a proxy property to refer to class iterable.

        Args:
            proxy (str): name of proxy property to create.
            attribute (str): name of attribute to link the proxy property to.
            default_value (Optional[Any]): default value to use when deleting
                an item in 'attribute'. Defaults to None.
            proxify_methods (Optiona[bool]): whether to create proxy methods
                replacing 'attribute' in the original method name with 'proxy'.
                So, for example, 'add_chapter' would become 'add_recipe' if
                'proxy' was 'recipe' and 'attribute' was 'chapter'. The original
                method remains as well as the proxy. Defaults to True.

        """

        self._attribute = attribute
        self._default_proxy_value = default_value
        self._proxify_attribute(proxy = proxy)
        if proxify_methods:
            self._proxify_methods(proxy = proxy)
        return self