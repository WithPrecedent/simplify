"""
.. module:: splitter
:synopsis: splits data into training, test, and/or validation sets
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.typesetter import SimpleDirector
from simplify.core.typesetter import Option


@dataclass
class Splitter(SimpleDirector):
    """Splits data into training, testing, and/or validation datasets.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    name: str = 'splitter'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self._options = SimpleOptions(options = {
            'group_kfold': Option(
                name = 'group_kfold',
                module = 'sklearn.model_selection',
                algorithm = 'GroupKFold',
                default = {'n_splits': 5},
                runtime = {'random_state': 'seed'},
                selected = True),
            'kfold': Option(
                name = 'kfold',
                module = 'sklearn.model_selection',
                algorithm = 'KFold',
                default = {'n_splits': 5, 'shuffle': False},
                runtime = {'random_state': 'seed'},
                selected = True),
            'stratified': Option(
                name = 'stratified',
                module = 'sklearn.model_selection',
                algorithm = 'StratifiedKFold',
                default = {'n_splits': 5, 'shuffle': False},
                runtime = {'random_state': 'seed'},
                selected = True),
            'time': Option(
                name = 'time',
                module = 'sklearn.model_selection',
                algorithm = 'TimeSeriesSplit',
                default = {'n_splits': 5},
                runtime = {'random_state': 'seed'},
                selected = True),
            'train_test': Option(
                name = 'train_test',
                module = 'sklearn.model_selection',
                algorithm = 'ShuffleSplit',
                default = {'test_size': 0.33},
                runtime = {'random_state': 'seed'},
                required = {'n_splits': 1},
                selected = True)}
        return self
