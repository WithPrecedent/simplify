
from dataclasses import dataclass
import re

from ..countertop import Countertop
from ..cookbook.steps import Custom
from ..cookbook.steps import Encoder
from ..cookbook.steps import Mixer
from ..cookbook.steps import Model
from ..cookbook.steps import Sampler
from ..cookbook.steps import Scaler
from ..cookbook.steps import Reducer
from ..cookbook.steps import Carver
from ..cookbook.steps import Splitter
from ..cookbook.recipe import Recipe


@dataclass
class Defrost(Countertop):

    file_path : str = ''
    recipes : object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    def _get_best_recipe(self):
        recipe_row = self.table[self.metrics[0]].argmax()
        recipe = self._parse_result(recipe_row)
        return recipe

    def _get_recipe(self, recipe_number = None, scorer = None):
        if recipe_number:
            recipe_row = self.table.iloc[recipe_number - 1]
        elif scorer:
            recipe_row = self.table[scorer].argmax()
        recipe = self._parse_result(recipe_row)
        return recipe

    def _get_all_recipes(self):
        recipes = []
        for row in self.table.iterrows():
            recipes.append(self._parse_result(self.table[row]))
        return recipes

    def _parse_ingredient(ingredient, return_cols = False):
        if ingredient == 'none':
            name = 'none'
            params = {}
        else:
            name = re.search('^\D*.?(?=\, parameters)', ingredient)[0]
            params = re.search('\{.*?\}', ingredient)[0]
        if return_cols:
            ingredient = re.sub('\{.*?\}', '')
            cols = re.search('\[.*', ingredient)[0]
            return name, params, cols
        else:
            return name, params

    def _parse_result(self, row):
        model = Model(self._parse_ingredient(row['model']))
        recipe = Recipe(row['recipe_number'],
                        order = row['ingredient_order'].split(),
                        scaler = Scaler(self._parse_ingredient(row['scaler'],
                                                    return_columns = True)),
                        splitter = Splitter(self._parse_ingredient(row['splitter'])),
                        encoder = Encoder(self._parse_ingredient(row['encoder'],
                                                    return_cols = True)),
                        interactor = Interactor(self._parse_ingredient(
                                row['interactor'], return_cols = True)),
                        splicer = Splicer(self._parse_ingredient(row['splicer'])),
                        sampler = Sampler(self._parse_ingredient(row['sampler'])),
                        custom = Custom(self._parse_ingredient(row['custom'])),
                        selector = Selector(self._parse_ingredient(row['selector'])),
                        model = model,
                        settings = self.settings)
        return recipe

    def implement(self):
        self.table = self.filer.load(self.file_path)
        return recipe