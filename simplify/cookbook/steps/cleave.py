
from dataclasses import dataclass

from .step import Step


@dataclass
class Cleave(Step):
    """Contains different groups of features (to allow comparison among them)
    used in the siMpLify package.
    """
    technique : str = 'none'
    parameters : object = None

    def __post_init__(self):
        super().__post_init__()
        self.techniques = {}
        self.algorithm = self.cleave
        return self

    def _prepare_cleaves(self):
        for group, columns in self.techniques.items():
            self.test_columns.extend(columns)
        if self.parameters['include_all']:
            self.techniques.update({'all' : self.test_columns})
        return self

    def cleave(self):
        pass
        return self

    def add_cleave(self, cleave_label, prefixes = [], columns = []):
        """For the cleavers in siMpLify, this step alows users to manually
        add a new cleave group to the cleaver dictionary.
        """
        temp_list = self.ingredients.create_column_list(prefixes = prefixes,
                                                        cols = columns)
        self.techniques.update({cleave_label : temp_list})
        return self

    def blend(self, ingredients):
        self._prepare_cleaves()
        if self.technique != 'none':
            if self.verbose:
                print('Testing', self.technique, 'predictors')
            if self.technique != 'all':
                cleave = self.techniques[self.technique]
                drop_list = [i for i in self.test_columns if i not in cleave]
                for col in drop_list:
                    if col in ingredients.x_train.columns:
                        ingredients.x_train.drop(col, axis = 'columns',
                                                 inplace = True)
                        ingredients.x_test.drop(col, axis = 'columns',
                                                inplace = True)
        return ingredients