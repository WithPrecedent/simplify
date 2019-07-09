
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import train_test_split

from .cookbook_step import CookbookStep


@dataclass
class Split(CookbookStep):
    """Splits data into training, testing, and/or validation sets or applies
    k-folds cross-validation.
    """
    technique : str = ''
    techniques : object = None
    parameters : object = None
    runtime_parameters : object = None
    auto_prepare : bool = True
    name : str = 'splitter'

    def __post_init__(self):
        self._set_defaults()
        super().__post_init__()
        return self

    def _cv_split(self):
        folder = StratifiedKFold(n_splits = self.kfolds,
                                 shuffle = False,
                                 random_state = self.seed)
        return folder

    def _no_split(self, ingredients):
        return ingredients

    def _one_split(self, x, y, split_size):
        x_train, x_test, y_train, y_test = (
                train_test_split(x, y,
                                 random_state = self.seed,
                                 test_size = split_size))
        return x_train, x_test, y_train, y_test

    def _set_defaults(self):
        if not self.techniques:
            self.techniques = {'time' : TimeSeriesSplit,
                               'train_test' : self._split_data,
                               'train_test_val' : self._split_data,
                               'cv' : self._cv_split,
                               'full' : self._no_split}
        self.default_parameters = {'test_size' : 0.33,
                                   'val_size' : 0,
                                   'kfolds' : 5,
                                   'krepeats' : 10}
        self.runtime_parameters = {'random_state' : self.seed}
        return self

    def _split_data(self, ingredients):
        (ingredients.x_train, ingredients.x_test, ingredients.y_train,
         ingredients.y_test) = (
                self._one_split(ingredients.x, ingredients.y,
                                self.parameters['test_size']))
        if 'val' in self.data_to_use:
            if self.val_size > 0:
                (ingredients.x_train, ingredients.x_val, ingredients.y_train,
                 ingredients.y_val) = (
                    self._one_split(ingredients.x_train, ingredients.y_train,
                                    self.parameters['val_size']))
            else:
                error = 'val_size must be > 0 if validation data selected.'
                raise ValueError(error)
        return ingredients

    def fit(self, ingredients):
        self.algorithm = self.techniques[self.technique]
        return self

    def fit_transform(self, ingredients):
        self.fit(ingredients)
        ingredients = self.transform(ingredients)
        return ingredients

    def prepare(self):
        """Adds parameters to algorithm."""
        self._check_parameters()
        self._select_parameters()
        self._check_runtime_parameters()
        if self.technique != 'none':
            self.algorithm = self.techniques[self.technique]
        return self

    def start(self, ingredients, recipe):
        ingredients = self.algorithm(ingredients)
        return ingredients

    def transform(self, ingredients):
        ingredients = self.transform(ingredients)
        return ingredients
