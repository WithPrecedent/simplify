"""
.. module:: states
:synopsis: state machines for siMpLify
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.utilities import listify


@dataclass
class SimpleState(ABC):
    """
    """
    def __post_init__(self) -> None:
        self.draft()
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
        return self

    def publish(self) -> str:
        return self.state


@dataclass
class DataState(SimpleState):
    """State machine for siMpLify project workflow.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        state (Optional[str]): initial state. Defaults to 'unsplit'.

    """
    name: Optional[str] = 'data_state_machine'
    state: Optional[str] = 'unsplit'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        # Sets possible states
        self.options = ['unsplit', 'train_test', 'train_val', 'full']
        return self


@dataclass
class Stage(SimpleState):
    """State machine for siMpLify project workflow.

    Args:
        idea (Idea): an instance of Idea.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    idea: 'Idea'
    name: Optional[str] = 'stage_machine'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _set_states(self) -> List[str]:
        """Determines list of possible stages from 'idea'.

        Returns:
            List[str]: states possible based upon user selections.

        """
        states = []
        for stage in listify(self.idea['simplify']['simplify_steps']):
            if stage == 'farmer':
                for step in self.idea['farmer']['farmer_steps']:
                    states.append(step)
            else:
                states.append(stage)
        return states

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Initializes state machine."""
        # Sets list of possible states based upon Idea instance options.
        self.options = self._set_states()
        # Sets initial state.
        self.state = self.options[0]
        return self