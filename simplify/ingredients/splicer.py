

from dataclasses import dataclass

from .ingredient import Ingredient


@dataclass
class Splicer(Ingredient):
    """Contains different groups of features (to allow comparison among them)
    used in the siMpLify package.
    """
    technique : str = 'none'
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {}
        self.algorithm = self.splice
        return self

    def _prepare_splices(self):
        for group, columns in self.options.items():
            self.test_columns.extend(columns)
        if self.params['include_all']:
            self.options.update({'all' : self.test_columns})
        return self

    def splice(self):
        pass
        return self

    def add_splice(self, splice_label, prefixes = [], columns = []):
        """
        For the splicers in siMpLify, this step alows users to manually
        add a new splice group to the splicer dictionary.
        """
        temp_list = self.codex.create_column_list(prefixes = prefixes,
                                                 cols = columns)
        self.options.update({splice_label : temp_list})
        return self

    def mix(self, codex):
        self._prepare_splices()
        if self.technique != 'none':
            if self.verbose:
                print('Testing', self.technique, 'predictors')
            if self.technique != 'all':
                splice = self.options[self.technique]
                drop_list = [i for i in self.test_columns if i not in splice]
                for col in drop_list:
                    if col in codex.x_train.columns:
                        codex.x_train.drop(col, axis = 'columns',
                                          inplace = True)
                        codex.x_test.drop(col, axis = 'columns', inplace = True)
        return codex