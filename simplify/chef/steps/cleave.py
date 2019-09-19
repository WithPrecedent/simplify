
from dataclasses import dataclass

from simplify.core.base import SimpleStep


@dataclass
class Cleave(SimpleStep):
    """Contains different groups of features (to allow comparison among them)
    used in the siMpLify package.
    """
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'cleaver'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        self.default_parameters = {}
        self.algorithm = self._cleave
        return self

    def _cleave(self, ingredients):
        if self.technique != 'all':
            cleave = self.options[self.technique]
            drop_list = [i for i in self.test_columns if i not in cleave]
            for col in drop_list:
                if col in ingredients.x_train.columns:
                    ingredients.x_train.drop(col, axis = 'columns',
                                             inplace = True)
                    ingredients.x_test.drop(col, axis = 'columns',
                                            inplace = True)
        return ingredients

    def _finalize_cleaves(self):
        for group, columns in self.options.items():
            self.test_columns.extend(columns)
        if self.parameters['include_all']:
            self.options.update({'all' : self.test_columns})
        return self

    def add(self, cleave_group, columns):
        """For the cleavers in siMpLify, this step alows users to manually
        add a new cleave group to the cleaver dictionary.
        """
        self.options.update({cleave_group : columns})
        return self

    def produce(self, ingredients, plan = None):
        if self.technique != 'none':
            self._finalize_cleaves()
            ingredients = self.algorithm(ingredients)
        return ingredients