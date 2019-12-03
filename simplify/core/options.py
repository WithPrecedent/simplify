"""
.. module:: options
:synopsis: base class for containing different options
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify


@dataclass
class SimpleOptions(MutableMapping):
    """Base class for different options to be stored.
    
    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.   
        choices (Optional[Dict[str, Any]]): alternative strategies stored
            in a dictionary in the following format:
                
                {str: Outline}

            If subclassing, 'choices' should be declared in the 'draft' method.
            Defaults to an empty dict.
        default_choices (Optional[Union[List[str], str]]): key(s) in 'choices' 
            to use if 'default' is selected. Defaults to an empty list. If
            not specified, and 'default' options are sought, all 'choices' will
            be returned.   
      
    """
    name: Optional[str] = None
    choices: Optional[Dict[str, Any]] = field(default_factory = dict())
    default_choices: Optional[Union[List[str], str]] = field(
        default_factory = list())
    
    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Sets wildcard values to check if a key doesn't exist in 'choices'.
        self.wildcards = {
            'all': self.all,
            'default': self.default,
            'defaults': self.default,
            'none': ['none'],
            'None': ['none']}
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in 'choices'.

        Args:
            item (str): name of key in 'choices'.

        """
        try:
            del self.choices[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> Any:
        """Returns item in 'choices'.
        
        If there are no matches, the method searches for a matching wildcard.

        Args:
            item (str): name of key in 'choices'.

        Raises:
            KeyError: if 'item' is not found in 'choices' and does not match
                a recognized wildcard.
            
        """
        try:
            return self.choices[item]
        except KeyError:
            try:
                return self.wildcards[item]
            except KeyError:
                raise KeyError(' '.join([item, 'is not in', self.name]))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets 'item' in 'choices' to 'value'.

        Args:
            item (str): name of key in 'choices'.
            value (Any): value to be paired with 'item' in 'choices'.

        """
        self.choices[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'choices'."""
        return iter(self.choices)

    def __len__(self) -> int:
        """Returns length of 'choices'."""
        return len(self.choices)

    """ Numeric Dunder Methods """

    def __add__(self, other: Union[Dict[str, Any], 'SimpleOptions']) -> None:
        """Combines two 'choices' dictionaries.

        Args:
            other (Union[Dict[str, Any],): either another 'SimpleOptions'
                instance or an 'choices' dict.

        Raises:
            TypeError: if 'other' is neither a 'SimpleOptions' instance nor
                a dict.

        """
        try:
            self.choices.update(other.choices)
        except AttributeError:
            try:
                self.choices.update(other)
            except AttributeError:
                raise TypeError(' '.join(
                    ['addition requires objects to be dict or SimpleOptions']))
        return self

    def __iadd__(self, other: Union[Dict[str, Any], 'SimpleOptions']) -> None:
        """Combines two 'choices' dictionaries.

        Args:
            other (Union[Dict[str, Any],): either another 'SimpleOptions'
                instance or an 'choices' dict.

        Raises:
            TypeError: if 'other' is neither a 'SimpleOptions' instance nor
                a dict.

        """
        self.__add__(other = other)
        return self

    def __invert__(self) -> None:
        """Reverses keys and values in 'choices'."""
        try:
            reversed = self.__reversed__()
            self.choices = reversed
        except AttributeError:
            self.choices = {}
        return self

    """ Sequence Dunder Methods """

    def __reversed__(self) -> Dict[Any, str]:
        """Returns 'choices' with keys and values reversed."""
        return {value: key for key, value in self.choices.items()}
 
    """ Core siMpLify Methods """

    def load(self, key: str) -> object:
        """Returns object from module based upon tuple in 'choices' value.
        
        Args:
            key (str): key to tuple of (module, object) to be loaded.
            
        Returns:
            object from module indicated in 'choices' value.
            
        """
        return self.choices['key'].load()
        
    def draft(self) -> None:
        """Subclasses should provide their own 'draft' methods."""
        return self
    
    def publish(self) -> None:
        """Sets 'default_choices' to all 'choices' if none exist."""   
        if not self.default_choices:
            self.default_choices = list(self.choices.keys())
        return self
    
    """ Properties """
     
    @property
    def all(self):
        return list(self.choices.keys())
    
    @property
    def default(self):
        return self.default_choices
    
    @default.setter
    def default(self, 
            choices: Union[str, List[str]], 
            override: Optional[bool]) -> None:
        if override or not self.default_choices:
            default_choices = listify(choices)
        else:
            default_choices.extend(listify(choices))
        return self