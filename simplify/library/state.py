"""
.. module:: state
:synopsis: base class for state management
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.library.utilities import listify


@dataclass
class SimpleState(object):
    """Base class for state management."""

    states: List[str]
    initial_state: Optional[str] = None
    
    def _post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns 'states' as an Iterable."""
        return iter(self.states)
    
    def __repr__(self) -> str:
        """Returns string name of 'state'."""
        return self.state

    def __str__(self) -> str:
        """Returns string name of 'state'."""
        return self.state

    """ State Management Methods """

    def change(self, new_state: str) -> None:
        """Changes 'state' to 'new_state'.

        Args:
            new_state(str): name of new state matching a string in 'states'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.states:
            self.state = new_state
        else:
            raise TypeError(' '.join([new_state, 'is not a recognized state']))

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates state machine default settings. """
        if self.initial_state:
            self.state = self.initial_state
        else:
            self.state = self.states[0]
        return self