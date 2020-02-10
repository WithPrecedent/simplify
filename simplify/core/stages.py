"""
.. module:: stages
:synopsis: state management made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import Container
from collections.abc import Iterable
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


@dataclass
class SimpleStages(object):
    """Base finite state machine for siMpLify.

    Args:
        parent (object): """

    parent: object
    stages: Optional[Union[List[str], Dict[str, 'SimpleStage']]] = field(
        default_factory = dict)
    initial: Optional[str] = None
    ordered: Optional[bool] = False

    def _post_init__(self) -> None:
        """Initializes class instance attributes."""
        self._create_stages()
        self._set_current()
        self._set_order()
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            stages: Optional[Union[
                'SimpleStages',
                List[str],
                Dict[str, 'SimpleStage']]] = None) -> 'SimpleStages':
        """

        """
        if isinstance(stages, SimpleStages):
            return stages
        elif isinstance(stages, (list, dict)):
            return cls(stages = stages)
        elif stages is None:
            return cls()
        else:
            raise TypeError('stages must be a SimpleStages, dict, or list')

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'current'."""
        return self.current

    def __str__(self) -> str:
        """Returns string name of 'current'."""
        return self.current

    """ Private Methods """

    # def _create_stages(self) -> None:
    #     return self

    def _set_current(self) -> None:
        """Sets current 'stage' upon initialization."""
        if self.initial and self.initial in self.stages:
            self.current = self.initial
        elif self.stages:
            self.current = self.stages[0]
        else:
            self.current = None
        self.previous = self.current
        return self

    def _set_order(self) -> None:
        if self.ordered:
            self.order = list(self.stages.keys())
        else:
            self.order = None
        return self

    """ Stage Management Methods """

    def add(self, name: str, stage: 'SimpleStage') -> None:

        return self

    def advance(self, instance: Optional[object] = None) -> object:
        self.previous = self.current
        current_index = self.order.index(self.current)
        self.current = self.order[self.current_index + 1]
        if instance is None:
            return self
        else:
            return self.current.apply(instance = instance)

    def change(self, new_stage: str) -> None:
        """Changes 'stage' to 'new_stage'.

        Args:
            new_stage(str): name of new stage matching a string in 'stages'.

        Raises:
            TypeError: if new_stage is not in 'stages'.

        """
        if new_stage in self.stages:
            self.previous = self.stage
            self.current = new_stage
            self.stages[self.current].apply(instance = self.parent)
        else:
            raise ValueError(' '.join([new_stage, 'is not a recognized stage']))


# def create_stages(
#         stages: Optional[Union[List[str], 'SimpleStage']]) -> 'SimpleStage':
#     """Creates a SimpleStage instance for state management.

#     Args:
#         stages (Optional[Union[List[str], 'SimpleStage']]): a 'SimpleStage'
#             instance or list of possible stages to create one. Defaults to None.
#             If no 'stages' are provided, a default list is used based upon the
#             default siMpLify Project structure.

#     Returns:
#         'SimpleStage': instance created with 'stages' argument or default
#         stages.

#     """
#     if not stages:
#         return SimpleStage(stages = [
#             'acquire',
#             'extract',
#             'munge',
#             'merge',
#             'finalize',
#             'analyze',
#             'summarize',
#             'criticize,
#             'visualize'])
#     elif isinstance(stages, list):
#         return SimpleStage(stages = stages)
#     elif isinstance(stages, SimpleStage):
#         return stages
#     else:
#         raise TypeError('stages must be list, None, or SimpleStage type.')


# @dataclass
# class SimpleStage(ABC):
#     """

#     """


#     """ Core siMpLify Methods """

#     @abstractmethod
#     def apply(self, instance: object) -> object:
#         return instance

# @dataclass
# class AnalyzerStage(SimpleStage):
#     """

#     """
#     __slots__ = ['train_set', 'test_set', 'import_folder', 'export_folder']
#     train_set: str
#     test_set: str

#     """ Core siMpLify Methods """

#     def apply(self, instance: object) -> object:
#         return instance