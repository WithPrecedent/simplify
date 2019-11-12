"""
.. module:: distributor
:synopsis: base class for distributor classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class SimpleDistributor(ABC):
    """Shares data between classes.

    It uses a visitor design pattern to inject needed attributes into classes
    throughout siMpLify.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """

    def __post_init__(self) -> None:
        self.draft()
        return self

    """ Private Methods """

    def _inject_attributes(self,
            attributes: Optional[Union[List[str], str]] = None,
            source: Optional[Union['SimpleComposite', str]] = None,
            target: Optional[Union['SimpleComposite', str]] = None) -> None:
        """Injects 'attributes' from 'source' to 'target'.

        Args:
            attributes (Optional[Union[List[str], str]]): attributes
                of 'source', if it exists, to inject into the current class.
                If none are provided, all possible shared are injected. Defaults
                to None.
            source (Optional[Union[SimpleComposite, str]]): source class for
                attributes to be taken. If none is provided, the method attempts
                to get them from 'planner'. If a str is passed, the source class
                is assumed to be a local attribute with that name. Defaults to
                None.
            target (Optional[Union[SimpleComposite, str]]): target class for
                attributes to be injected. If none is provided, the current
                class instance is used. If a str is passed, the target class is
                assumed to be a local attribute with that name. Defaults to
                None.

        """
        if attributes is None:
            attributes = ['idea', 'depot']
        if source is None:
            try:
                source = self.planner
            except TypeError:
                source = self
        else:
            try:
                source = getattr(self, source)
            except TypeError:
                pass
            try:
                target = getattr(self, target)
            except TypeError:
                pass
        for attribute in listify(attributes):
            try:
                setattr(target, attribute, getattr(source, attribute))
                if attribute == 'idea':
                    self._inject_idea()
            except TypeError:
                pass
        return self


    """ Public Methods """

    def inject_planner(self,
            attributes: Union[List[Union['SimpleComposite', str]], str]) -> None:
        """Adds 'attributes' to 'planner'.

        Args:
            attributes (Union[List[Union[str, SimpleComposite]], str]): attribute(s)
                to be injected.

        """
        self._inject_attributes(
            attributes = attributes,
            source = self,
            target = self.planner)
        return self

    def inject_shared(self,
            attributes: Union[List[Union['SimpleComposite', str]], str]) -> None:
        """Adds 'attributes' to 'shared'.

        Args:
            attributes (Union[List[Union[str, SimpleComposite]], str]): attribute(s)
                to be injected.

        """
        for shared in self.shared.values():
            self._inject_attributes(
                attributes = attributes,
                source = self,
                target = shared)
        return self

    def inject_techniques(self,
            attributes: Union[List[Union['SimpleComposite', str]], str]) -> None:
        """Adds 'attributes' to 'techniques'.

        Args:
            attributes (Union[List[Union[str, SimpleComposite]], str]): attribute(s)
                to be injected.

        """
        for technique in self.techniques.values():
            self._inject_attributes(
                attributes = attributes,
                source = self,
                target = technique)
        return self

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        return self

    @abstractmethod
    def publish(self, instance: 'SimpleComposite') -> None:
        setattr(instance, self.name, self)
        return self