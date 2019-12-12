"""
.. module:: manuscript
:synopsis: composite tree abstract base class
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.options import ManuscriptOptions
from simplify.core.utilities import listify


@dataclass
class Manuscript(ABC):
    """Base class for data processing, analysis, and visualization.

    Manuscript contains the shared methods, properties, and attributes for a
    modified composite tree pattern for organizing the various subpackages in
    siMpLify.

    Args:
        steps (Optional[List[str], str, Dict[str, str]]): ordered list of
            steps to use with each item matching a key in 'options', a single
            item which matchings a key in 'options' or a dictionary where each
            key matches a key in options and each value is a 'technique'
            parameter to be sent to a child class. Defaults to an empty dict.
        options (Optional[Union['ManuscriptOptions', Dict[str, Any]]]): allows
            setting of 'options' property with an argument. Defaults to None.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. Defaults to True.

    """

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns '_children' dictionary as iterable."""
        return iter(self._children)
    
    """ Private Methods """
    
    def _draft_techniques(self) -> List[str]:
        """Tries to get techniques from shared Idea instance.

        Returns:
            List[str] of techniques to use.
        """
        techniques = None
        try:
            techniques = listify(
                self.options.idea['_'.join([self.name, 'techniques'])])
        except AttributeError:
            try:
                techniques = listify(
                    self.options.idea['_'.join([self.technique, 'techniques'])])
            except AttributeError:
                pass
        return techniques

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Required method that sets default values."""
         # Injects attributes from Idea instance, if values exist.
        self = self.options.idea.apply(instance = self)
        # initializes class options.
        self._draft_options()
        # Initializes class steps.
        self._draft_steps()
        return self

    def publish(self) -> None:
        """Finalizes objects in 'options' for application."""
        self.options.publish(steps = self.steps)
        return self

    def apply(self, data: Optional[object], **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (object): data object for methods to be applied.

        """
        if data is None:
            try:
                data = self.ingredients
            except AttributeError:
                pass
        for step, technique in self.steps.items():
            data = self.options[technique].options.apply(
                key = technique,
                data = data,
                **kwargs)
        return data

    """ Composite Methods and Properties """

    def add_children(self, children: Union[List[str], str]) -> None:
        """Adds outline(s) to '_children' from 'options' based on key(s).
        Args:
            keys (Union[List[str], str]): key(s) to 'options'.
        """
        self._children.extend(listify(children))
        return self

    def load_child(self, file_path: str) -> None:
        """Imports a single child from disk and adds it to the class iterable.

        Args:
            file_path (str): a path where the file to be loaded is located.

        """
        self.add_children(children =
            self.options.inventory.load(
                file_path = file_path,
                file_format = 'pickle'))
        return self

    @property
    def parent(self) -> 'Manuscript':
        """Returns '_parent' attribute."""
        return self._parent

    @parent.setter
    def parent(self, parent: 'Manuscript') -> None:
        """Sets '_parent' attribute to 'parent' argument.
        Args:
            parent (Manuscript): Manuscript class up one level in
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
    def children(self) -> Dict[str, Union['Outline', 'Manuscript']]:
        """Returns '_children' attribute.
        Returns:
            Dict of str access keys and Outline or Manuscript values.
        """
        return self._children

    @children.setter
    def children(self, children: Dict[str, 'Outline']) -> None:
        """Assigns 'children' to '_children' attribute.
        If 'override' is False, 'children' are added to '_children'.
        Args:
            children (Dict[str, 'Outline']): dictionary with str for reference
                keys and values of 'Manuscript'.
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
            options: Union['ManuscriptOptions', Dict[str, Any]]) -> None:
        """Assigns 'options' to '_options' attribute.

        Args:
            options (options: Union['ManuscriptOptions', Dict[str, Any]]): either
                another 'ManuscriptOptions' instance or an options dict.

        """
        self.options.add(options = options)
        return self

    @property
    def options(self) -> 'ManuscriptOptions':
        """Returns '_options' attribute."""
        return self.options

    @options.setter
    def options(self, options: Union['ManuscriptOptions', Dict[str, Any]]) -> None:
        """Assigns 'options' to '_options' attribute.

        Args:
            options (Union['ManuscriptOptions', Dict[str, Any]]): ManuscriptOptions
                instance or a dictionary to be stored within a ManuscriptOptions
                instance (this should follow the form outlined in the
                ManuscriptOptions documentation).

        """
        if isinstance(options, dict):
            self.options = ManuscriptOptions(options = options)
        else:
            self.options.add(options = options)
        return self

    @options.deleter
    def options(self, options: Union[List[str], str]) -> None:
        """ Removes 'options' from '_options' attribute.

        Args:
            options (Union[List[str], str]): key(s) to options classes to
                remove from '_options'.

        """
        for option in listify(options):
            try:
                del self.options[option]
            except KeyError:
                pass
        return self