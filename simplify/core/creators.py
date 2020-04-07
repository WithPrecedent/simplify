"""
.. module:: base
:synopsis: project workflow made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import abc
import collections.abc
import dataclasses
import importlib
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import pathos.multiprocessing as mp
except ImportError:
    import multiprocessing as mp


@dataclasses.dataclass
class SimpleHandler(abc.ABC):
    """Base class for creating or modifying other siMpLify classes."""

    """ Required Subclass Methods """

    @abc.abstractmethod
    def apply(self, data: 'SimpleFoundation', **kwargs) -> 'SimpleFoundation':
        """Subclasses must provide their own methods."""
        return self


@dataclasses.dataclass
class SimpleParallel(SimpleHandler):
    """Applies workflow using one or more CPU or GPU cores.

    Args:
        gpu (Optional[bool]): whether to use GPU cores, when possible, to
            parallelize operations (True) or to solely use CPU cores (False).
            Defaults to False.

    """

    gpu: Optional[bool] = False

    """ Private Methods """

    def _apply_gpu(self, process: Callable, data: object, **kwargs) -> object:
        """

        """
        try:
            return process(data, **kwargs)
        except TypeError:
            return self._apply_cpu(process = process, data = data, **kwargs)

    def _apply_cpu(self, process: Callable, data: object, **kwargs) -> object:
        """

        """
        results = []
        arguments = data
        arguments.update(kwargs)
        with mp.Pool() as pool:
            results.append(pool.starmap(method, arguments))
        pool.close()
        return results

    """ Core siMpLify Methods """

    def apply(self, process: Callable, data: object, **kwargs) -> object:
        """

        """
        if self.gpu:
            return self._apply_gpu(process = process, data = data, **kwargs)
        else:
            return self._apply_cpu(process = process, data = data, **kwargs)
        return results


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
            if (self._attribute in item
                    and not item.startswith('__')
                    and callabe(item)):
                self.__dict__[item.replace(self._attribute, proxy)] = (
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

