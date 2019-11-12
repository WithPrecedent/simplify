"""
.. module:: splitter
:synopsis: splits data into training, test, and/or validation sets
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleComposer
from simplify.core.technique import SimpleDesign


@dataclass
class Splitter(SimpleComposer):
    """Splits data into training, testing, and/or validation datasets.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'splitter'

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self.options = {
            'group_kfold': SimpleDesign(
                name = 'group_kfold',
                module = 'sklearn.model_selection',
                algorithm = 'GroupKFold',
                default = {'n_splits': 5},
                runtime = {'random_state': 'seed'},
                selected = True),
            'kfold': SimpleDesign(
                name = 'kfold',
                module = 'sklearn.model_selection',
                algorithm = 'KFold',
                default = {'n_splits': 5, 'shuffle': False},
                runtime = {'random_state': 'seed'},
                selected = True),
            'stratified': SimpleDesign(
                name = 'stratified',
                module = 'sklearn.model_selection',
                algorithm = 'StratifiedKFold',
                default = {'n_splits': 5, 'shuffle': False},
                runtime = {'random_state': 'seed'},
                selected = True),
            'time': SimpleDesign(
                name = 'time',
                module = 'sklearn.model_selection',
                algorithm = 'TimeSeriesSplit',
                default = {'n_splits': 5},
                runtime = {'random_state': 'seed'},
                selected = True),
            'train_test': SimpleDesign(
                name = 'train_test',
                module = 'sklearn.model_selection',
                algorithm = 'ShuffleSplit',
                default = {'test_size': 0.33},
                runtime = {'random_state': 'seed'},
                required = {'n_splits': 1},
                selected = True)}
        return self
