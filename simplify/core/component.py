"""
.. module:: component
:synopsis: project structure made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import abc
import dataclasses
import importlib
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


@dataclasses.dataclass
class SimpleComponent(abc.ABC):
    """Base class for components in a 'SimpleSystem'.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().

    """
    name: Optional[str] = None

    def __post_init__(self) -> None:
        """Sets 'name' to default value if it is not passed."""
        self.name = self.name or self.__class__.__name__.lower()
        return self


@dataclasses.dataclass
class SimpleLoader(SimpleComponent):
    """Base class for lazy loaders for low-level siMpLify objects.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        default_module (Optional[str]): a backup name of module where object to
            use is located (can either be a siMpLify or non-siMpLify module).
            Defaults to 'simplify.core'.

    """
    name: Optional[str] = None
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.core')
    default_module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.core')

    """ Public Methods """

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
            object: from 'module' or 'default_module'.

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
                    raise ImportError(
                        f'{getattr(self, attribute)} is neither in \
                        {self.module} nor {self.default_module}')
        # If 'attribute' is not a string, it is returned as is.
        else:
            return getattr(self, attribute)


@dataclasses.dataclass
class SimplePlan(SimpleComponent, collections.abc.MutableMapping):
    """Base class for iterating over lists of 'SimpleComponent' instances.

    A 'SimplePlan' stores a list of items with 'name' attributes. Each 'name'
    acts as a key to create the facade of a dictionary with the items in the
    stored list serving as values. This allows for duplicate keys and the
    storage of class instances at the expense of lookup speed. Since normal
    use cases do not include repeat accessing of 'SimplePlan' instances, the
    loss of lookup speed should have negligible effect.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        contents (Optional[List[SimpleComponent]]): stored list. Defaults to an
            empty list.

    """
    name: Optional[str] = None
    contents: Optional[List[SimpleComponent]] = dataclasses.field(
        default_factory = list)

    """ Public Methods """

    def add(self,
            contents: Union[
                'SimplePlan',
                List['SimpleComponent'],
                'SimpleComponent']) -> None:
        """Combines arguments with 'contents'.

        Args:
            contents (Union['SimplePlan', Dict[str, Any]]): another 'SimplePlan'
                instance or compatible dictionary.

        """
        if isinstance(contents, SimplePlan):
            self.contents.extend(contents.contents)
        elif isinstance(contents, SimpleComponent):
            self.contents.append(contents)
        elif isinstance(contents, list):
            self.contents.extend(contents)
        else:
            raise TypeError(
                f'contents must be a SimpleComponent, SimplePlan, or list type')
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> List[SimpleComponent]:
        """Returns value(s) for 'key' in 'contents'.

        The method searches for 'all', 'default', and 'none' matching wildcard
        options before searching for direct matches in 'contents'.

        Args:
            key (Union[List[str], str]): name(s) of key(s) in 'contents'.

        Returns:
            Union[List[Any], Any]: value(s) stored in 'contents'.

        """
        return [item for item in self.contents if item.name == key]

    def __setitem__(self, key: Union[str], value: 'SimpleComponent') -> None:
        """Adds 'value' to 'contents' if 'key' matches 'value.name'.

        Args:
            key (str): name of key(s) to set in 'contents'.
            value ('SimpleComponent'): value(s) to be added at the end of
                'contents'.

        """
        if hasattr(value, name) and value.name in [key]:
            self.add(contents = contents)
        else:
            raise TypeError(
                f'{self.name} requires a value with a name atttribute')
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (str): name(s) of key(s) in 'contents' to
                delete the key/value pair.

        """
        try:
            self.contents = [item for item in self.contents if item.name != key]
        except AttributeError:
            raise TypeError(
                f'{self.name} requires a value with a name atttribute')
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
            other: Union[
                'SimplePlan',
                List['SimpleComponent'],
                'SimpleComponent']) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union['SimplePlan', Dict[str, Any]]): another 'SimplePlan'
                instance or compatible dictionary.

        """
        self.add(contents = other)
        return self

    def __iadd__(self,
            other: Union[
                'SimplePlan',
                List['SimpleComponent'],
                'SimpleComponent']) -> None:
        """Combines argument with 'contents'.

        Args:
            other (Union['SimplePlan', Dict[str, Any]]): another 'SimplePlan'
                instance or compatible dictionary.

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
        """Returns representation of 'contents'.

        Returns:
            str: representation of 'contents'.

        """
        return self.contents.__str__()


@dataclasses.dataclass
class SimpleRepository(SimpleComponent, collections.abc.MutableMapping):
    """Base class for policy and option storage.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.

    """
    name: Optional[str] = None
    contents: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory = dict)
    defaults: Optional[List[str]] = dataclasses.field(default_factory = list)

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super().__post_init__()
        # Stores nested dictionaries as 'SimpleRepository' instances.
        self.contents = self._nestify(contents = self.contents)
        # Sets 'default' to all keys of 'contents', if not passed.
        self.defaults = self.defaults or list(self.contents.keys())
        return self

    """ Public Methods """

    def add(self, contents: Union['SimpleRepository', Dict[str, Any]]) -> None:
        """Combines arguments with 'contents'.

        Args:
            contents (Union['SimpleRepository', Dict[str, Any]]): another
                'SimpleRepository' instance/subclass or a compatible dictionary.

        """
        self.contents.update(contents)
        self.contents = self._nestify(contents = self.contents)
        return self

    def subset(self, subset: Union[Any, List[Any]]) -> 'SimpleRepository':
        """Returns a subset of 'contents'.

        Args:
            subset (Union[Any, List[Any]]): key(s) to get key/value pairs from
                'dictionary'.

        Returns:
            'SimpleRepository': with only keys in 'subset'.

        """
        return self.__class__(
            name = name,
            contents = utilities.subsetify(
                dictionary = self.contents,
                subset = subset),
            defaults = self.defaults)

    """ Required ABC Methods """

    def __getitem__(self, key: Union[List[str], str]) -> Union[List[Any], Any]:
        """Returns value(s) for 'key' in 'contents'.

        The method searches for 'all', 'default', and 'none' matching wildcard
        options before searching for direct matches in 'contents'.

        Args:
            key (Union[List[str], str]): name(s) of key(s) in 'contents'.

        Returns:
            Union[List[Any], Any]: value(s) stored in 'contents'.

        """
        if key in ['all', ['all']]:
            return list(self.contents.values())
        elif key in ['default', ['default']]:
            return list(utilities.subsetify(
                dictionary = self.contents,
                subset = self.defaults).values())
        elif key in ['none', ['none'], '', ['']]:
            return []
        else:
            try:
                return self.contents[key]
            except TypeError:
                try:
                    return [self.contents[k] for k in key if k in self.contents]
                except KeyError:
                    raise KeyError(f'{key} is not in {self.name}')
            except KeyError:
                raise KeyError(f'{key} is not in {self.name}')

    def __setitem__(self,
            key: Union[List[str], str],
            value: Union[List[Any], Any]) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (Union[List[str], str]): name of key(s) to set in 'contents'.
            value (Union[List[Any], Any]): value(s) to be paired with 'key' in
                'contents'.

        """
        if key in ['default', ['default']]:
            self.defaults = value
        else:
            try:
                self.contents[key] = value
            except TypeError:
                self.contents.update(dict(zip(key, value)))
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
        return f'{self.name}, contents: {self.contents.__str__()}, \
            defaults: {self.defaults}'

    """ Private Methods """

    def _nestify(self,
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
        new_repository = self.__new__()
        for key, value in contents.items():
            if isinstance(value, dict):
                new_repository.add(
                    contents = {key: self._nestify(contents = value)})
            else:
                new_repository.add(contents = {key: value})
        return new_repository


@dataclasses.dataclass
class SimpleProxy(abc.ABC):
    """Mixin which creates proxy name for an instance attribute.

    The 'proxify' method dynamically creates a property to access the stored
    attribute. This allows class instances to customize names of stored
    attributes while still using base siMpLify classes.

    """

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
            if (self._attribute in item
                    and not item.startswith('__')
                    and callabe(item)):
                self.__dict__[item.replace(self._attribute, proxy)] = (
                    getattr(self, item))
        return self