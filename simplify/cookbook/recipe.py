
from dataclasses import dataclass


@dataclass
class Recipe(object):
    """Defines rules for analyzing data in the siMpLify Cookbook subpackage.

    Attributes:
        techniques: a list of techniques containing the classes to be used at
            each stage of a Recipe.
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
    """
    techniques : object = None
    name : str = 'recipe'

    def __post_init__(self):
        pass
        return self

    def __call__(self, *args, **kwargs):
        """When called as a function, a Recipe class or subclass instance will
        return the start method.
        """
        return self.start(*args, **kwargs)

    def _check_attributes(self):
        """Checks if corresponding attribute exists for every item in the
        self.techniques list.
        """
        for technique in self.techniques:
            if not hasattr(self, technique):
                error = technique + ' has not been passed to Recipe class.'
                raise AttributeError(error)
        return self

    def prepare(self):
        """Prepares instance of Recipe."""
        self._check_attributes()
        # Creates a boolean attribute as to whether the validation set is being
        # used for later access by a Critic instance.
        if 'val' in self.data_to_use:
            self.val_set = True
        else:
            self.val_set = False
        return self

    def start(self, ingredients):
        """Applies the Recipe methods to the passed ingredients."""
        techniques = self.techniques.copy()
        self.ingredients = ingredients
        # noinspection PyProtectedMember
        self.ingredients._remap_dataframes(data_to_use = self.data_to_use)
        self.ingredients.split_xy(label = self.label)
        for technique in self.techniques:
            techniques.remove(technique)
            if technique != 'splitter':
                self.ingredients = getattr(self, technique).start(
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
           for technique in techniques:
                self.ingredients = getattr(self, technique).start(
                        self.ingredients, self)
        return self