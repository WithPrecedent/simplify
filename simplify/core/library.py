"""
.. module:: manuscripts
:synopsis: base class for storing a group of manuscripts
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.manuscript import Manuscript


@dataclass
class Library(MutableMapping):
    """Stores built siMpLify objects.
    
    This class is used to prevent unnecessay duplication of object creation and
    to reduce overall memory usage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. Defaults to 'library'.

    """
    name: Optional[str] = 'library'
    manuscripts: Optional[Dict[str, Dict[str, 'Manuscript']]] = field(
        default_factory = dict)

    def __post_init__(self):
        """Calls initialization methods and sets class instance defaults."""
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __delitem__(self, item: str) -> None:
        """Deletes item in 'manuscripts'.

        Args:
            item (str): name of key in 'manuscripts'.

        """
        try:
            del self.manuscripts[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: str) -> 'Manuscript':
        """Returns item in 'manuscripts'.

        Args:
            item (str): name of key in 'manuscripts'.

        Returns:
            'Manuscript' instance.

        Raises:
            KeyError: if 'item' is not found in 'manuscripts'.

        """
        try:
            return self.manuscripts[item]
        except KeyError:
            raise KeyError(' '.join([item, 'is not in', self.name]))

    def __setitem__(self, item: str, value: Any) -> None:
        """Sets 'item' in 'manuscripts' to 'value'.

        Args:
            item (str): name of key in 'manuscripts'.
            value (Any): value to be paired with 'item' in 'manuscripts'.

        """
        self.manuscripts[item] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'manuscripts'."""
        return iter(self.manuscripts)

    def __len__(self) -> int:
        """Returns length of 'manuscripts'."""
        return len(self.manuscripts)

    """ Other Dunder Methods """

    def __add__(self, 
            other: Union['Library', 
                Dict[str, Dict[str, 'Manuscript']]]) -> None:
        """Adds Manuscript instances to 'manuscripts'.

        Args:
            other (Union['Library', Dict[tuple[str, str], 'Manuscript']]): 
                either another 'Library' instance or a dictionary compatible
                with 'Library'.

        Raises:
            TypeError: if 'other' is neither a 'Library' instance nor a dict.

        """
        self.add(manuscripts = other)
        return self

    def __iadd__(self, 
            other: Union['Library', 
                Dict[str, Dict[str, 'Manuscript']]]) -> None:
        """Adds Manuscript instances to 'manuscripts'.

        Args:
            other (Union['Library', Dict[tuple[str, str], 'Manuscript']]): 
                either another 'Library' instance or a dictionary compatible
                with 'Library'.

        Raises:
            TypeError: if 'other' is neither a 'Library' instance nor a dict.
                
        """
        self.add(manuscripts = other)
        return self

    """ Public Methods """

    def add(self, 
            manuscripts: Union['Library', 
                Dict[str, Dict[str, 'Manuscript']]]) -> None:
        """Adds Manuscript instances to 'manuscripts'.

        Args:
            manuscripts (Union['Library', Dict[tuple[str, str], 'Manuscript']]): 
                either another 'Library' instance or a dictionary compatible
                with 'Library'.

        Raises:
            TypeError: if 'other' is neither a 'Library' instance nor a dict.
                
        """
        try:
            self.manuscripts.update(manuscripts.manuscripts)
        except AttributeError:
            try:
                self.manuscripts.update(manuscripts)
            except AttributeError:
                raise TypeError(' '.join(
                    ['addition requires dict or Library types']))
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        return self

    def apply(self, data: object = None, **kwargs) -> None:
        """Calls 'apply' method for published option matching 'technique'.

        Args:
            data (object): object for option to be applied. Defaults
                to None.
            kwargs: any additional parameters to pass to the option's 'apply'
                method.

        """
        for key, manuscript in self.manuscripts.items():
            manuscript.apply(data = data, **kwargs)
        return self