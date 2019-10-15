"""
.. module:: split
:synopsis: splits data into training, test, and/or validation sets
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

from simplify.core.technique import SimpleTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'group_kfold': ['sklearn.model_selection', 'GroupKFold'],
    'kfold': ['sklearn.model_selection', 'KFold'],
    'stratified': ['sklearn.model_selection', 'StratifiedKFold'],
    'time': ['sklearn.model_selection', 'TimeSeriesSplit'],
    'train_test': ['sklearn.model_selection', 'ShuffleSplit']}


@dataclass
class Split(SimpleTechnique):
    """Splits data into training, testing, and/or validation sets, uses time
    series splits, or applies k-folds cross-validation.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
            
    """

    technique: object = None
    parameters: object = None
    name: str = 'split'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.default_parameters = {
            'train_test': {'test_size': 0.33},
            'kfold': {'n_splits': 5, 'shuffle': False},
            'stratified': {'n_splits': 5, 'shuffle': False},
            'group_kfold': {'n_splits': 5},
            'time': {'n_splits': 5}}
        self.extra_parameters = {'train_test': {'n_splits': 1}}
        self.selected_parameters = True
        return self

    def publish(self):
        self.runtime_parameters = {'random_state': self.seed}
        super().publish()
        return self
