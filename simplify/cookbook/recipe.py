
import copy
from dataclasses import dataclass

from ..managers import Plan


@dataclass
class Recipe(Plan):
    """Defines rules for analyzing data in the siMpLify Cookbook subpackage.

    Attributes:
        steps: dictionary of steps containing the name of the step and
            corresponding classes. Dictionary keys and values should be placed
            in order that they should be completed.
        number: counter of recipe, used for file and folder naming.

    """
    steps : object = None
    name : str = 'recipe'
    structure : str = 'compare'

    def __post_init__(self):
        super().__post_init__()
        return self

    def prepare(self):
        super().prepare()
        if 'val' in self.data_to_use:
            self.val_set = True
        else:
            self.val_set = False
        return self

    def start(self, ingredients):
        """Applies the Recipe methods to the passed ingredients."""
        steps = self.steps.copy()
        self.ingredients = ingredients
        self.ingredients._remap_dataframes(data_to_use = self.data_to_use)
        self.ingredients.split_xy(label = self.label)
        for step in self.steps:
            steps.remove(step)
            if step != 'splitter':
                self.ingredients = getattr(self, step).start(
                        self.ingredients, self)
            else:
                break
        for train_index, test_index in self.splitter.algorithm.split(
                self.ingredients.x, self.ingredients.y):
           self.ingredients.x_train, self.ingredients.x_test = (
                   self.ingredients.x.iloc[train_index],
                   self.ingredients.x.iloc[test_index])
           self.ingredients.y_train, self.ingredients.y_test = (
                   self.ingredients.y.iloc[train_index],
                   self.ingredients.y.iloc[test_index])
           for step in steps:
                self.ingredients = getattr(self, step).start(
                        self.ingredients, self)
        return self