from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from .step import Step


@dataclass
class Splitter(Step):

    name : str = 'none'
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'time' : TimeSeriesSplit,
                        'train_test' : self._split_data,
                        'train_test_val' : self._split_data,
                        'cv' : self._cv_split,
                        'full' : self._no_split}
        self.defaults = {'test_size' : 0.33,
                         'val_size' : 0,
                         'kfolds' : 5,
                         'krepeats' : 10}
        return self

    def _cv_split(self, data):
#        for train_index, test_index in self.params['folder'].split(data.x, data.y):
#            data.x_train, data.x_test = (data.x.iloc[train_index],
#                                         data.x.iloc[test_index])
#            data.y_train, data.y_test = (data.y.iloc[train_index],
#                                         data.y.iloc[test_index])
        return data

    def _split_data(self, data):
        data.x_train, data.x_test, data.y_train, data.y_test = (
                self._one_split(data.x, data.y, self.params['test_size']))
        if 'val' in self.data_to_use:
            if self.val_size > 0:
                data.x_train, data.x_val, data.y_train, data.y_val = (
                    self._one_split(data.x_train, data.y_train,
                                    self.params['val_size']))
            else:
                error = 'val_size must be > 0 if validation data selected.'
                raise ValueError(error)
        return data

    def _one_split(self, x, y, split_size):
        x_train, x_test, y_train, y_test = (
                train_test_split(x, y,
                                 random_state = self.seed,
                                 test_size = split_size))
        return x_train, x_test, y_train, y_test

    def _no_split(self, data):
        return data

    def mix(self, data):
        if self.name != 'none':
            if self.verbose:
                print('Splitting data')
            self.runtime_params = {'random_state' : self.seed}
            self._check_params()
            self.params.update(self.runtime_params)
            self.algorithm = self.options[self.name]
        return self.algorithm(data)

    def fit(self, data):
        self.algorithm = self.options[self.name]
        return self

    def transform(self, data):
        return self.algorithm(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)