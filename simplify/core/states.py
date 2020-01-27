"""
.. module:: states
:synopsis: state managament made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from collections.abc import Iterable
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from functools import update_wrapper
from functools import wraps
from importlib import import_module
from inspect import signature
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


@dataclass
class SimpleState(Container):
    """Base class for state management."""

    states: List[str]
    initial_state: Optional[str] = None

    def _post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Automatically calls 'draft' method.
        self.draft()
        return self

    """ Required ABC Methods """

    def __contains__(self, attribute: str) -> bool:
        """Returns whether 'attribute' exists in 'states'.

        Args:
            attribute (str): name of state to check.

        Returns:
            bool: whether the attribute exists in 'states'.

        """
        return attribute in self.states

    """ Other Dunder Methods """

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
            new_state(str): name of new state matching a string in 'states'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.states:
            self.previous_state = self.state
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
        self.previous_state = self.state
        return self

    def publish(self) -> None:
        """Returns string name of 'state'."""
        return self.state

    def apply(self, instance: object) -> object:
        """Injects 'state' and 'previous_state' into 'instance.'

        Args:
            instance (object): object to add state attributes to.

        Returns:
            object: with state attributes added.

        """
        instance.state = self.state
        instance.previous_state = self.previous_state
        return instance


def create_states(
        states: Optional[Union[List[str], 'SimpleState']]) -> 'SimpleState':
    """Creates a SimpleState instance for state management.

    Args:
        states (Optional[Union[List[str], 'SimpleState']]): a 'SimpleState'
            instance or list of possible states to create one. Defaults to None.
            If no 'states' are provided, a default list is used based upon the
            default siMpLify Project structure.

    Returns:
        'SimpleState': instance created with 'states' argument or default
        states.

    """
    if not states:
        return SimpleState(states = [
            'sow',
            'reap',
            'clean',
            'bale',
            'deliver',
            'analyst',
            'actuary',
            'critic',
            'artist'])
    elif isinstance(states, list):
        return SimpleState(states = states)
    elif isinstance(states, SimpleState):
        return states
    else:
        raise TypeError('states must be list, None, or SimpleState type.')

