
from dataclasses import dataclass

from sklearn.model_selection import (GroupKFold, KFold, ShuffleSplit,
                                     StratifiedKFold, TimeSeriesSplit)

from simplify.core.base import SimpleStep


@dataclass
class Split(SimpleStep):
    """Splits data into training, testing, and/or validation sets, uses time 
    series splits, or applies k-folds cross-validation.
    
    Args:
        technique(str): name of technique - it should always be 'gauss'
        parameters(dict): dictionary of parameters to pass to selected technique
            algorithm.
        auto_finalize(bool): whether 'finalize' method should be called when the
            class is instanced. This should generally be set to True.
        store_names(bool): whether this class requires the feature names to be
            stored before the 'finalize' and 'produce' methods or called and
            then restored after both are utilized. This should be set to True
            when the class is using numpy methods.
        name(str): name of class for matching settings in the Idea instance and
            for labeling the columns in files exported by Critic.
    """
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    store_names : bool = False
    name : str = 'splitter'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
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

    def finalize(self):
        self._nestify_parameters()
        self._finalize_parameters()
        if self.technique in ['train_test']:
            self.parameters.update({'n_splits' : 1})
        if self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
        return self