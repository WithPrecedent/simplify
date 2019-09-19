
from dataclasses import dataclass

from .models.classifier import Classifier
from .models.clusterer import Clusterer
from .models.regressor import Regressor
from simplify.core.base import SimpleStep


@dataclass
class Model(SimpleStep):
    """Applies machine learning algorithms based upon user selections."""
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'model'

    def __post_init__(self):
        self.idea_sections = ['cookbook']
        super().__post_init__()
        return self

    def draft(self):
        self.options = {'classifier' : Classifier,
                        'clusterer' : Clusterer,
                        'regressor' : Regressor}
        self.runtime_parameters = {'random_state' : self.seed}
        self.checks = ['idea']
        return self

    def finalize(self):
        """Adds parameters to machine learning algorithm."""
        if self.technique != 'none':
            if not hasattr(self, 'parameters') or not self.parameters:
                self.parameters = self.idea[self.technique]
            self._finalize_parameters()
            self.algorithm = self.options[self.model_type](
                    technique = self.technique,
                    parameters = self.parameters)
            self.algorithm.finalize()
        return self

    def fit_transform(self, x, y):
        error = 'fit_transform is not implemented for machine learning models'
        raise NotImplementedError(error)

    def produce(self, ingredients, plan = None):
        """Applies model from recipe to ingredients data."""
        if self.technique != 'none':
            if self.verbose:
                print('Applying', self.technique, 'model')
            ingredients = self.algorithm.produce(ingredients = ingredients)
        return ingredients

    def transform(self, x, y):
        error = 'transform is not implemented for machine learning models'
        raise NotImplementedError(error)

