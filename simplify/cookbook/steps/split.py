
from dataclasses import dataclass

from sklearn.model_selection import (GroupKFold, KFold, ShuffleSplit,
                                     StratifiedKFold, TimeSeriesSplit)

from ..cookbook_step import CookbookStep


@dataclass
class Split(CookbookStep):
    """Splits data into training, testing, and/or validation sets or applies
    k-folds cross-validation.
    """
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'splitter'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        self.options = {'group_kfold' : GroupKFold,
                        'kfold' : KFold,
                        'stratified' : StratifiedKFold,
                        'time' : TimeSeriesSplit,
                        'train_test' : ShuffleSplit}
        if self.technique in ['train_test']:
            self.default_parameters = {'test_size' : 0.33}
        elif self.technique in ['kfold', 'stratified']:
            self.default_parameters = {'n_splits' : 5,
                                       'shuffle' : False}
        elif self.technique in ['group_kfold', 'time']:
            self.default_parameters = {'n_splits' : 5}
        self.runtime_parameters = {'random_state' : self.seed}
        self.selected_parameters = True
        return self

    def prepare(self):
        """Adds parameters to algorithm."""
        self._check_parameters()
        self._select_parameters()
        if self.technique in ['train_test']:
            self.parameters.update({'n_splits' : 1})
        if self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
        return self