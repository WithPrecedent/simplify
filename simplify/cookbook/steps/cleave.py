
from dataclasses import dataclass

from .step import Step


@dataclass
class Cleave(Step):
    """Contains different groups of features (to allow comparison among them)
    used in the siMpLify package.
    """

    technique : str = 'none'
    techniques : object = None
    parameters : object = None
    runtime_parameters : object = None
    data_to_use : str = 'train'
    name : str = 'cleaver'

    def __post_init__(self):
        self.techniques = {}
        self.algorithm = self._cleave
        return self

    def _cleave(self, ingredients):
        if self.technique != 'all':
            cleave = self.techniques[self.technique]
            drop_list = [i for i in self.test_columns if i not in cleave]
            for col in drop_list:
                if col in ingredients.x_train.columns:
                    ingredients.x_train.drop(col, axis = 'columns',
                                             inplace = True)
                    ingredients.x_test.drop(col, axis = 'columns',
                                            inplace = True)
        return self

    def _prepare_cleaves(self):
        for group, columns in self.techniques.items():
            self.test_columns.extend(columns)
        if self.parameters['include_all']:
            self.techniques.update({'all' : self.test_columns})
        return self

    def add(self, cleave_group, columns):
        """For the cleavers in siMpLify, this step alows users to manually
        add a new cleave group to the cleaver dictionary.
        """
        self.techniques.update({cleave_group : columns})
        return self

    def implement(self, ingredients):
        self._prepare_cleaves()
        if self.technique != 'none':
            self.algorithm(ingredients)
        return ingredients
