
from dataclasses import dataclass

from ..implements.tools import listify

@dataclass
class Plan(object):
    """Defines rules for sowing, reaping, cleaning, bundling, and delivering
    data as part of the siMpLify Almanac subpackage.

    Attributes:
        techniques: a list of Almanac step techniques to complete.
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
    """
    techniques : object = None
    name : str = 'plan'
    index_column : str = 'index_universal'

    def __post_init__(self):
        if not self.techniques:
            self.techniques = []
        return self
#
#    def _check_attributes(self):
#        for technique in self.techniques:
#            if not hasattr(self, technique.technique):
#                error = technique.technique + ' has not been passed to class.'
#                raise AttributeError(error)
#        return self
#
#    def _set_columns(self, variable):
#        if (not hasattr(self, 'column_names')
#            and variable.technique in ['organize']):
#            self.column_names = listify(self.index_column)
#            self.column_names.extend(variable.columns)
#        elif not hasattr(self, 'column_names'):
#            self.column_names = list(variable.columns.keys())
#        return self

    def _start_file(self, ingredients):
        with open(
                self.inventory.path_in, mode = 'r', errors = 'ignore',
                encoding = self.menu['files']['file_encoding']) as a_file:
            ingredients.source = a_file.read()
            for technique in self.techniques:
                ingredients = technique.start(ingredients = ingredients)
            self.inventory.save(variable = ingredients.df)
        return ingredients

    def _start_glob(self, ingredients):
        self.inventory.initialize_writer(
                file_path = self.inventory.path_out)
        ingredients.create_series()
        for file_num, a_path in enumerate(self.inventory.path_in):
            if (file_num + 1) % 100 == 0 and self.verbose:
                print(file_num + 1, 'files parsed')
            with open(
                    a_path, mode = 'r', errors = 'ignore',
                    encoding = self.menu['files']['file_encoding']) as a_file:
                ingredients.source = a_file.read()
                print(ingredients.df)
                ingredients.df[self.index_column] = file_num + 1
                for technique in self.techniques:
                    ingredients = technique.start(ingredients = ingredients)
                self.inventory.save(variable = ingredients.df)
        return ingredients

    def prepare(self):
#        self._check_attributes()
#        for technique in self.techniques:
#            self._set_columns(variable = technique)
        return self

    def start(self, ingredients):
        """Applies the Almanac technique classes to the passed ingredients."""
        if isinstance(self.inventory.path_in, list):
            ingredients = self._start_glob(ingredients = ingredients)
        else:
            ingredients = self._start_file(ingredients = ingredients)
        return ingredients