"""
Splkcer is a class containing different groups of features (to allow comparison
among them) used in the siMpLify package.
"""

from dataclasses import dataclass

from simplify.step import Step

@dataclass
class Splicer(Step):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {}
        self.algorithm = self.splice
        return self

    def __getitem__(self, value):
        """
        If user wants to test different combinations of features ("splices"),
        this Step returns a list of possible splicers set by user.
        """
        if self.options:
            if self.params['include_all']:
                test_columns = []
                for group, columns in self.options.items():
                    test_columns.extend(columns)
                self.options.update({'all' : test_columns})
            splicers = list(self.data.splice_options.keys())
        else:
            splicers = ['none']
        return splicers

    def splice(self):
        return self

    def add_splice(self, splice, prefixes = [], columns = []):
        """
        For the splicers in siMpLify, this step alows users to manually
        add a new splice group to the splicer dictionary.
        """
        temp_list = self.data.create_column_list(prefixes = prefixes,
                                                 cols = columns)
        self.options.update({splice : temp_list})
        return self

    def fit(self, x, y):
        if self.params['include_all']:
            test_columns = []
            for group, columns in self.options.items():
                test_columns.extend(columns)
            self.options.update({'all' : test_columns})
        return self

    def transform(self, x):
        drop_list = [i for i in self.test_columns if i not in self.Step]
        for col in drop_list:
            if col in x.columns:
                x.drop(col, axis = 'columns', inplace = True)
        return x