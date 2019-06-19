
from dataclasses import dataclass

from .step import Step

@dataclass
class Custom(Step):

    technique : str = 'none'
    name : str = 'custom'
    techniques : object = None
    parameters : object = None
    runtime_parameters : object = None
    data_to_use : str = 'train'

    def __post_init__(self):
        self.options = {'train' : self._blend_train,
                        'train_test' : self._blend_train_test,
                        'full' : self._blend_full}
        self._add_algorithms()
        return self

    def _add_algorithms(self):
        if self.techniques:
            for technique, algorithm in self.techniques.items():
                setattr(self, technique, algorithm)
        return self

    def _set_defaults(self):
        if not self.runtime_parameters:
            self.runtime_parameters = {}
        if not self.parameters:
            self.parameters = {}
        return self

    def _blend_full(self, ingredients):
        if self.technique != 'none':
            self._set_defaults()
            self._initialize()
            ingredients.x, ingredients.y = self.fit_transform(
                    ingredients.x, ingredients.y)
        return ingredients

    def _blend_train(self, ingredients):
        if self.technique != 'none':
            self._set_defaults()
            self._initialize()
            ingredients.x_train, ingredients.y_train = self.fit_transform(
                    ingredients.x_train, ingredients.y_train)
        return ingredients

    def _blend_train_test(self, ingredients):
        if self.technique != 'none':
            self._set_defaults()
            self._initialize()
            ingredients.x_train, ingredients.y_train = self.fit_transform(
                    ingredients.x_train, ingredients.y_train)
            ingredients.x_test, ingredients.y_test = self.fit_transform(
                    ingredients.x_test, ingredients.y_test)
        return ingredients

    def blend(self, ingredients):
        ingredients = self.options[self.data_to_use](ingredients)
        return ingredients