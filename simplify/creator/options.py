"""
.. module:: options
:synopsis: base class for storing different options
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.library.utilities import listify
from simplify.library.utilities import proxify


@dataclass
class Options(MutableMapping):
    """Base class for different options to be stored.

    The CodexOptions class should be injected with the shared Idea instance before it 
    or any subclass is instanced. This is done automatically through the normal 
    siMpLify Project access point. But if you are creating a completely 
    customized workflow, this step must be taken for siMpLify to work properly.

    Args:
        options (Optional[Dict[str, Any]]): alternative strategies stored
            in a dictionary in the following format:

                {str: Outline}

            If subclassing, 'drafted' should be declared in the 'draft' method.
            Defaults to an empty dict.
        default_options (Optional[Union[List[str], str]]): key(s) to use if the
            'default' property is selected. Defaults to an empty list. If not 
            specified, and 'default' options are sought, all options will be 
            returned.
            
    """
    options: Optional[Dict[str, Any]] = field(default_factory = dict)
    default_options: Optional[Union[List[str], str]] = field(
        default_factory = list)
    parent: Optional[object] = None
    
    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        # Sets private 'codex' attribute.
        self._parent = parent
        # Applies proxy attribute names, if any are set.
        try:
            self = proxify(instance = self, proxies = self.proxies)
        except AttributeError:
            pass
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in options.

        Args:
            item (str): name of key in options.

        """
        try:
            del getattr(self, self.state)[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> Any:
        """Returns item in options.

        If there are no matches, the method searches for a matching wildcard.

        Args:
            item (str): name of key in options.

        Raises:
            KeyError: if 'item' is not found in options and does not match
                a recognized wildcard.

        """
        try:
            return getattr(self, self.state)[item]
        except KeyError:
            try:
                return self.wildcards[item]
            except KeyError:
                raise KeyError(' '.join([item, 'is not in', self.name]))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets 'item' in options to 'value'.

        Args:
            item (str): name of key in options.
            value (Any): value to be paired with 'item' in options.

        """
        getattr(self, self.state)[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of options."""
        return iter(getattr(self, self.state))

    def __len__(self) -> int:
        """Returns length of options."""
        return len(getattr(self, self.state))

    """ Other Dunder Methods """

    def __add__(self, other: Union[Dict[str, Any], 'CodexOptions']) -> None:
        """Combines two options dictionaries.

        Args:
            other (Union[Dict[str, Any],): either another 'CodexOptions'
                instance or an options dict.

        Raises:
            TypeError: if 'other' is neither a 'CodexOptions' instance nor
                a dict.

        """
        try:
            getattr(self, self.state).update(getattr(other, self.state))
        except AttributeError:
            try:
                getattr(self, self.state).update(other)
            except AttributeError:
                raise TypeError(' '.join(
                    ['addition requires objects to be dict or CodexOptions']))
        return self

    def __iadd__(self, other: Union[Dict[str, Any], 'CodexOptions']) -> None:
        """Combines two options dictionaries.

        Args:
            other (Union[Dict[str, Any],): either another 'CodexOptions'
                instance or an options dict.

        Raises:
            TypeError: if 'other' is neither a 'CodexOptions' instance nor
                a dict.

        """
        self.__add__(other = other)
        return self

    def __invert__(self) -> None:
        """Reverses keys and values in options."""
        try:
            setattr(self, self.state, self.__reversed__())
        except AttributeError:
            setattr(self, self.state, {})
        return self

    def __reversed__(self) -> Dict[Any, str]:
        """Returns options with keys and values reversed."""
        return {value: key for key, value in getattr(self, self.state).items()}

    """ Relational Properties """
    
    @property
    def parent(self) -> None:
        return self._parent

    @parent.setter
    def parent(self, parent: object) -> None:
        self._parent = parent
        return self

    @parent.deleter
    def parent(self) -> None:
        self._parent = None
        return self            

         
@dataclass
class CodexOptions(Options):
    """Base class for SimpleCodex subclasses with 'options'.

    Args:
        options (Optional[Dict[str, Any]]): alternative strategies stored
            in a dictionary in the following format:

                {str: Outline}

            If subclassing, 'drafted' should be declared in the 'draft' method.
            Defaults to an empty dict.
        default_options (Optional[Union[List[str], str]]): key(s) to use if the
            'default' property is selected. Defaults to an empty list. If not 
            specified, and 'default' options are sought, all options will be 
            returned.
            
    """
    options: Optional[Dict[str, Any]] = field(default_factory = dict)
    default_options: Optional[Union[List[str], str]] = field(
        default_factory = list)
    parent: Optional[object] = None

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        # Sets wildcard values to check if a key doesn't exist in options.
        self.wildcards = {
            'all': self.all,
            'default': self.default,
            'defaults': self.default,
            'none': ['none'],
            'None': ['none']}
        self.proxies = {'parent': 'codex'}
        self.state = 'drafted'
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def publish(self,
            steps: Dict[str, str], 
            data: Optional[object] = None) -> None:
        """Loads and instances selected options.

        Args:
            steps (steps: Dict[str, str]): key(s) to options that
                are to be used in a siMpLify project. Only the selected
                'techniques' will be lazily loaded into memory and instanced.
            data (Optional[object]): an object to pass when an options instance
                is published. Defaults to None.

        """
        # Sets state for access methods.
        self.state = 'published'
        # Instances and publishes all selected options.
        for step, technique in steps.items():
            # Lazily loads all stored options from stored Outline instances.
            option = self.drafted[step].load()
            if step == technique:
                instance = option()
            else:
                instance = option(technique = technique)
            instance.publish(data = data)
            self.published[step] = instance
        return self

    def apply(self,
            technique: str,
            data: Optional[object] = None,
            **kwargs) -> object:
        """Calls 'apply' method for published option matching 'technique'.

        Args:
            technique (str): technique for specific option to be applied.
            data (Optional[object]): object for option to be applied. Defaults
                to None.
            kwargs: any additional parameters to pass to the option's 'apply'
                method.

        Returns:
            object is returned if data is passed, otherwise None is returned.

        """
        # Sets state for access methods.
        self.state = 'applied'
        data = self.published[technique].apply(data = data)
        self.applied[technique] = self.published[technique]
        return data

    """ Properties """

    @property
    def all(self):
        return list(self.drafted.keys())

    @property
    def default(self):
        return self.default_options or self.all

    @default.setter
    def default(self, options: Union[str, List[str]]) -> None:
        self.default_options = listify(options)
        return self

    @default.deleter
    def default(self, options: Union[str, List[str]]) -> None:
        for option in listify(options):
            try:
                del self.default_options[option]
            except KeyError:
                pass
        return self