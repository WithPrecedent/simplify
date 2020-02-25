"""
.. module:: states
:synopsis: state managament made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""
from abc import ABC
from abc import abstractmethod
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
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

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
            'explorer',
            'critic',
            'artist'])
    elif isinstance(states, list):
        return SimpleState(states = states)
    elif isinstance(states, SimpleState):
        return states
    else:
        raise TypeError('states must be list, None, or SimpleState type.')


# @dataclass
# class SimpleStates(object):
#     """Base finite state machine for siMpLify."""

#     states: Optional[List[str]] = None
#     initial: Optional[str] = None

#     def _post_init__(self) -> None:
#         """Initializes class instance attributes."""
#         self._set_states()
#         self.order = list(self.states.keys())
#         return self

#     """ Factory Method """

#     @classmethod
#     def create(cls,
#             states: Optional[Union[
#                 List[str], 'DataStates']] = None) -> 'DataStates':
#         """

#         """
#         if isinstance(states, DataStates):
#             return states
#         elif isinstance(states, (list, dict)):
#             return cls(states = states)
#         elif states is None:
#             return cls()
#         else:
#             raise TypeError('states must be a DataStates, dict, or list')

#     """ Dunder Methods """

#     def __repr__(self) -> str:
#         """Returns string name of 'current'."""
#         return self.current

#     def __str__(self) -> str:
#         """Returns string name of 'current'."""
#         return self.current

#     """ Private Methods """

#     def _set_states(self) -> None:
#         self.defaults = {
#             'raw': DataState(
#                 default: 'full',
#                 import_folder: 'raw',
#                 export_folder: 'raw'),
#             'interim': DataState(
#                 default: 'full',
#                 import_folder: 'raw',
#                 export_folder: 'interim'),
#             'processed': DataState(
#                 default: 'full',
#                 import_folder: 'interim'),
#             'staging': DataState(
#                 default: 'x',
#                 training: ('x', 'y'),
#                 testing: ('x', 'y')),
#             'testing': DataState(
#                 training: ('x_train', 'y_train'),
#                 testing: ('x_test', 'y_test')),
#             'validating': DataState(
#                 training: ('x_train', 'y_train'),
#                 testing: ('x_val', 'y_val')))}
#         if self.states is None:
#             self.states = self.defaults
#         elif isinstance(self.states, list):
#             new_states = {}
#             for state in self.states:
#                 new_states[state] = self.defaults[state]
#             self.states = new_states
#         elif isinstance(self.states, DataStates):
#             self.states = self.states.states
#         if self.initial:
#             self.current = self.initial
#         else:
#             self.current = self.states[0]
#         self.previous = self.state
#         return self

#     """ State Management Methods """

#     def add(self, name: str, state: 'SimpleState') -> None:

#         return self

#     def advance(self, instance: Optional[object] = None) -> object:
#         self.previous = self.current
#         current_index = self.order.index(self.current)
#         self.current = self.order[self.current_index + 1]
#         if instance is None:
#             return self
#         else:
#             return self.current.apply(instance = instance)


# @dataclass
# class SimpleState(ABC):
#     """

#     """
#     __slots__ = []

#     """ Core siMpLify Methods """

#     @abstractmethod
#     def apply(self, instance: object) -> object:
#         return instance

# @dataclass
# class DataState(object):
#     """

#     """
#     __slots__ = ['train_set', 'test_set', 'import_folder', 'export_folder']
#     train_set: str
#     test_set: str
#     import_folder: str
#     export_folder: str
