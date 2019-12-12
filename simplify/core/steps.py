"""
.. module:: steps
:synopsis: base class for iterable steps
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableSequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify


@dataclass
class SimpleSteps(MutableSequence):

    steps: List[str] = field(default_factory = list)
    related: 'SimpleCodex' = None

    """ Required Dunder and Public Methods """

    def __delitem__(self, item: Union[str, int]) -> None:
        """Deletes item in 'steps'.

        Args:
            item (Union[str, int]): name or index in 'steps'.

        """
        try:
            del self.steps[item]
        except KeyError:
            pass
        return self

    def __getitem__(self, item: Union[str, int]) -> Any:
        """Returns item in 'steps'.

        Args:
            item (Union[str, int]): name or index of key in 'steps'.

        Returns:
            Any: item stored in 'steps'.

        Raises:
            KeyError: if 'item' is not found in 'steps'.

        """
        try:
            return self.steps.index(item)
        except TypeError:
            try:
                return self.steps[item]
            except KeyError:
                raise KeyError(' '.join(
                    [item, 'is not in', self.related.name, 'steps']))

    def __setitem__(self, index: int, item: str) -> None:
        """Adds 'item' to 'steps' at 'index' location.

        Args:
            index (int): location for 'item' to be inserted.
            item (str): item to be inserted.

        """
        self.steps[index] = item
        return self

    def __len__(self) -> int:
        """Returns length of 'steps'."""
        return len(self.steps)

    def insert(self, index: int, item: str) -> None:
        """Inserts 'item' in 'steps' at the beginning.

        Args:
            item (str): item to be inserted.

        """
        self.steps.insert(index, item)
        return self