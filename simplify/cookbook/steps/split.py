
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from .step import Step


@dataclass
class Split(Step):

    technique : str = 'none'
    parameters : object = None

    def __post_init__(self):
        super().__post_init__()
        self.techniques = {'time' : TimeSeriesSplit,
                           'train_test' : self._split_data,
                           'train_test_val' : self._split_data,
                           'cv' : self._cv_split,
                           'full' : self._no_split}
        self.defaults = {'test_size' : 0.33,
                         'val_size' : 0,
                         'kfolds' : 5,
                         'krepeats' : 10}
        return self

    def _cv_split(self, codex):
#        for train_index, test_index in self.parameters['folder'].split(codex.x, codex.y):
#            codex.x_train, codex.x_test = (codex.x.iloc[train_index],
#                                         codex.x.iloc[test_index])
#            codex.y_train, codex.y_test = (codex.y.iloc[train_index],
#                                         codex.y.iloc[test_index])
        return codex

    def _split_data(self, codex):
        codex.x_train, codex.x_test, codex.y_train, codex.y_test = (
                self._one_split(codex.x, codex.y,
                                self.parameters['test_size']))
        if 'val' in self.data_to_use:
            if self.val_size > 0:
                codex.x_train, codex.x_val, codex.y_train, codex.y_val = (
                    self._one_split(codex.x_train, codex.y_train,
                                    self.parameters['val_size']))
            else:
                error = 'val_size must be > 0 if validation data selected.'
                raise ValueError(error)
        return codex

    def _one_split(self, x, y, split_size):
        x_train, x_test, y_train, y_test = (
                train_test_split(x, y,
                                 random_state = self.seed,
                                 test_size = split_size))
        return x_train, x_test, y_train, y_test

    def _no_split(self, codex):
        return codex

    def blend(self, codex):
        if self.technique != 'none':
            if self.verbose:
                print('Splitting data')
            self.runtime_parameters = {'random_state' : self.seed}
            self._check_parameters()
            self.parameters.update(self.runtime_parameters)
            self.algorithm = self.techniques[self.technique]
        return self.algorithm(codex)

    def fit(self, codex):
        self.algorithm = self.techniques[self.technique]
        return self

    def transform(self, codex):
        return self.algorithm(codex)

    def fit_transform(self, codex):
        self.fit(codex)
        return self.transform(codex)