
from dataclasses import dataclass

from .models.classifier import Classifier
from .models.clusterer import Clusterer
from .models.regressor import Regressor
from ..cookbook_step import CookbookStep


@dataclass
class Model(CookbookStep):
    """Applies machine learning algorithms based upon user selections."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'model'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _define(self):
        self.options = {'classifier' : Classifier,
                        'clusterer' : Clusterer,
                        'regressor' : Regressor}
        self.runtime_parameters = {'random_state' : self.seed}
        return self

    def fit_transform(self, x, y):
        error = 'fit_transform is not implemented for machine learning models'
        raise NotImplementedError(error)

    def prepare(self):
        """Adds parameters to machine learning algorithm."""
        if self.technique != 'none':
            if not hasattr(self, 'parameters') or not self.parameters:
                self.parameters = self.menu[self.technique]
            self._check_parameters()
            self.algorithm = self.options[self.model_type](
                    technique = self.technique,
                    parameters = self.parameters)
            self.algorithm.prepare()
        return self

    def start(self, ingredients, recipe):
        """Applies model from recipe to ingredients data."""
        if self.technique != 'none':
            if self.verbose:
                print('Applying', self.technique, 'model')
            ingredients = self.algorithm.start(ingredients = ingredients)
        return ingredients

    def transform(self, x, y):
        error = 'transform is not implemented for machine learning models'
        raise NotImplementedError(error)