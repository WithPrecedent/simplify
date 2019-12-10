"""
.. module:: codex
:synopsis: composite tree abstract base class
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.creator.options import Options
from simplify.library.utilities import listify
from simplify.library.utilities import proxify


@dataclass
class SimpleCodex(ABC):
    """Base class for data processing, analysis, and visualization.

    SimpleCodex contains the shared methods, properties, and attributes for a
    modified composite tree pattern for organizing the various subpackages in
    siMpLify.

    Args:
        steps (Optional[List[str], str, Dict[str, str]]): ordered list of
            steps to use with each item matching a key in 'options', a single
            item which matchings a key in 'options' or a dictionary where each
            key matches a key in options and each value is a 'technique'
            parameter to be sent to a child class. Defaults to an empty dict.
        options (Optional[Union['Options', Dict[str, Any]]]): allows
            setting of 'options' property with an argument. Defaults to None.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. Defaults to True.

    """
    steps: Optional[List[str], str, Dict[str, str]] = field(
        default_factory = dict)
    options: (Optional[Union['Options', Dict[str, Any]]]) = None
    auto_publish: optional[bool] = True

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

    def _draft_options(self) -> None:
        """Subclasses should provide their own methods to create 'options'.

        If the subclasses also allow for passing of '_options', the code below
        should be included as well.Any

        """
        if not hasattr(self, '_options'):
            self._options = self.options
        if self.options is None:
            self.options = Options(options = {}, author = self)
        elif isinstance(self.options, Dict):
            self.options = Options(options = self.options, author = self)
        return self

    def _draft_steps(self) -> None:
        """If 'steps' does not exist, gets 'steps' from 'idea'.

        If there are no matching 'steps' or 'steps' in 'idea', a list with
        'none' is created for 'steps'.

        """

        if not self.steps:
            try:
                self.steps = self.options.idea['_'.join([self.name, 'steps'])]
            except AttributeError:
                pass
        if not self.steps:
            try:
                self.steps = self.options.idea[
                    '_'.join([self.name, 'steps'])]
            except AttributeError:
                pass
        self._technique_parameter = False
        self.steps = listify(self.steps, use_null = True) or []
        self.steps = listify(self.steps, use_null = True) or []
        if len(self.steps) == len(self.steps) and len(self.steps) > 0:
            self._technique_parameter = True
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Required method that sets default values."""
        # initializes class options.
        self._draft_options()
        # Adds proxy attributes if 'proxies' has been set.
        try:
            self = proxify(instance = self, proxies = self.proxies)
        except AttributeError:
            pass
        # Injects attributes from Idea instance, if values exist.
        self = self.idea.apply(instance = self)
        # Initializes class steps.
        self._draft_steps()
        return self

    def publish(self, data: Optional[object] = None) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' method.

        Args:
            data (Optional[object]): an optional object needed for the method.

        """
        if data is None:
            try:
                data = self.ingredients
            except AttributeError:
                pass
        self.options.publish(
            steps = self.steps,
            steps = self.steps,
            data = data)
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
        for technique in self.steps:
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
        self.children.append(
            self.filer.load(
                file_path = file_path,
                file_format = 'pickle'))
        return self

    @property
    def parent(self) -> 'SimpleCodex':
        """Returns '_parent' attribute."""
        return self._parent

    @parent.setter
    def parent(self, parent: 'SimpleCodex') -> None:
        """Sets '_parent' attribute to 'parent' argument.
        Args:
            parent (SimpleCodex): SimpleCodex class up one level in
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
    def children(self) -> Dict[str, Union['Outline', 'SimpleCodex']]:
        """Returns '_children' attribute.
        Returns:
            Dict of str access keys and Outline or SimpleCodex values.
        """
        return self._children

    @children.setter
    def children(self, children: Dict[str, 'Outline']) -> None:
        """Assigns 'children' to '_children' attribute.
        If 'override' is False, 'children' are added to '_children'.
        Args:
            children (Dict[str, 'Outline']): dictionary with str for reference
                keys and values of 'SimpleCodex'.
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
            options: Union['Options', Dict[str, Any]]) -> None:
        """Assigns 'options' to '_options' attribute.

        Args:
            options (options: Union['Options', Dict[str, Any]]): either
                another 'Options' instance or an options dict.

        """
        self.options += options
        return self

    @property
    def options(self) -> 'Options':
        """Returns '_options' attribute."""
        return self.options

    @options.setter
    def options(self, options: Union['Options', Dict[str, Any]]) -> None:
        """Assigns 'options' to '_options' attribute.

        Args:
            options (Union['Options', Dict[str, Any]]): Options
                instance or a dictionary to be stored within a Options
                instance (this should follow the form outlined in the
                Options documentation).

        """
        if isinstance(options, dict):
            self.options = Options(options = options)
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