"""
.. module:: state
:synopsis: state machine made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import MutableSequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.utilities import listify



@dataclass
class State(SimpleSequence):
    """Base class for state management."""

    states: Optional[List[str]] = field(default_factory = list)
    state: Optional[str] = None
    related: Optional[object] = None

    def _post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        self.sequence = self.states
        self.super().__post_init__()
        return self

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'state'."""
        return self.state

    def __str__(self) -> str:
        """Returns string name of 'state'."""
        return self.state

    """ State Management Methods """

    def change(self, new_state: Optional[str] = None) -> None:
        """Changes 'state' to 'new_state'.

        Args:
            new_state(str): name of new state matching a string in 'sequence'.

        Raises:
            TypeError: if new_state is not in 'sequence'.
        """
        if new_state is None:
            new_state = self.next(self.sequence)
            self.state = new_state
        elif new_state in self.sequence:
            self.state = new_state
        else:
            raise KeyError(' '.join([new_state, 'is not a recognized state']))

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates state machine default settings."""
        if not self.state:
            try:
                self.state = self.sequence[0]
            except TypeError:
                self.state = None
        return self

    def publish(self) -> str:
        """Returns string name of 'state'."""
        return self.state