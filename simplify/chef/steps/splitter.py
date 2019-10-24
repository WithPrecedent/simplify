"""
.. module:: splitter
:synopsis: splits data into training, test, and/or validation sets
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.chef.composer import ChefAlgorithm as Algorithm
from simplify.chef.composer import ChefComposer as Composer
from simplify.chef.composer import ChefTechnique as Technique


@dataclass
class Splitter(Composer):
    """Splits data into training, testing, and/or validation datasets.
    """

    name: str = 'splitter'

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self):
        self.group_kfold = Technique(
            name = 'group_kfold',
            module = 'sklearn.model_selection',
            algorithm = 'GroupKFold',
            defaults = {'n_splits': 5},
            runtimes = {'random_state': 'seed'},
            selected = True)
        self.kfold = Technique(
            name = 'kfold',
            module = 'sklearn.model_selection',
            algorithm = 'KFold',
            defaults = {'n_splits': 5, 'shuffle': False},
            runtimes = {'random_state': 'seed'},
            selected = True)
        self.stratified = Technique(
            name = 'stratified',
            module = 'sklearn.model_selection',
            algorithm = 'StratifiedKFold',
            defaults = {'n_splits': 5, 'shuffle': False},
            runtimes = {'random_state': 'seed'},
            selected = True)
        self.time = Technique(
            name = 'time',
            module = 'sklearn.model_selection',
            algorithm = 'TimeSeriesSplit',
            defaults = {'n_splits': 5},
            runtimes = {'random_state': 'seed'},
            selected = True)
        self.train_test = Technique(
            name = 'train_test',
            module = 'sklearn.model_selection',
            algorithm = 'ShuffleSplit',
            defaults = {'test_size': 0.33},
            runtimes = {'random_state': 'seed'},
            extras = {'n_splits': 1},
            selected = True)
        super().draft()
        return self
