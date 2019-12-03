"""
.. module:: state
:synopsis: base class for state management
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.base.options import SimpleOptions
from simplify.core.utilities import listify


@dataclass
class SimpleState(SimpleOptions, ABC):
    """Base class for state management."""

    def _post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Calls SimpleOptions __post_init__ method.
        SimpleOptions.__post_init__()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'state'."""
        return self.publish()

    def __str__(self) -> str:
        """Returns string name of 'state'."""
        return self.publish()

    """ State Management Methods """

    def change(self, new_state: str) -> None:
        """Changes 'state' to 'new_state'.

        Args:
            new_state(str): name of new state matching a string in 'options'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.options:
            self.state = new_state
        else:
            error = ' '.join([new_state, 'is not a recognized state'])
            raise TypeError(error)

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        """Creates state machine default settings.

        Subclass instances should provide their own methods.

        """
        return self

    def publish(self) -> str:
        """Returns current state in 'state' attribute."""
        return self.state
