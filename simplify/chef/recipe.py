"""
.. module:: recipe
:synopsis: stores steps for data analysis and machine learning
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.iterable import SimpleIterable


@dataclass
class Recipe(SimpleIterable):
    """Contains steps for analyzing data in the siMpLify Cookbook subpackage.

    Args:
        number(int): number of recipe in a sequence - used for recordkeeping
            purposes.
        steps(dict): dictionary containing keys of SimpleTechnique names
            (strings) and values of SimpleIterable subclass instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    number: int = 0
    steps: object = None
    name: str = 'recipe'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Private Methods """

    def _calculate_hyperparameters(self):
        """Computes hyperparameters that can be determined by the source data
        (without creating data leakage problems).

        This method currently only support xgboost's scale_pos_weight
        parameter. Future hyperparameter computations will be added as they
        are discovered.
        """
        # 'ingredients' attribute is required before method can be called.
        if self.ingredients is not None:
            # Data is split in oder for certain values to be computed that
            # require features and the label to be split.
            self.ingredients.split_xy(label = self.label)
            # Model class is injected with scale_pos_weight for algorithms that
            # use that parameter.
            self.options['model'].scale_pos_weight = (
                    len(self.ingredients.y.index) /
                    ((self.ingredients.y == 1).sum())) - 1
        return self

    """ Public Import/Export Methods """

    def save(self, file_path = None, folder = None, file_name = None):
        self.depot.save(variable = self,
                        file_path = file_path,
                        folder = folder,
                        file_name = file_name,
                        file_format = 'pickle')
        return

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
                'scale': ['simplify.chef.steps.scale', 'Scale'],
                'split': ['simplify.chef.steps.split', 'Split'],
                'encode': ['simplify.chef.steps.encode', 'Encode'],
                'mix': ['simplify.chef.steps.mix', 'Mix'],
                'cleave': ['simplify.chef.steps.cleave', 'Cleave'],
                'sample': ['simplify.chef.steps.sample', 'Sample'],
                'reduce': ['simplify.chef.steps.reduce', 'Reduce'],
                'model': ['simplify.chef.steps.model', 'Model']}
        self.iterator = 'steps'
        self.iterable_setting = 'chef_steps'
        return self

    def implement(self, ingredients):
        """Applies the recipe steps to the passed ingredients."""
        steps = getattr(self, 'iterable').copy()
        self.ingredients = ingredients
        self.ingredients.split_xy(label = self.label)
        # If using cross-validation or other data splitting technique, the
        # pre-split methods apply to the 'x' data. After the split, steps
        # must incorporate the split into 'x_train' and 'x_test'.
        for step, technique in getattr(self, 'iterable').items:
            del getattr(self, 'iterable')[step]
            if step == 'split':
                break
            else:
                self.ingredients = self.steps[step].implement(
                    ingredients = self.ingredients,
                    plan = self)
        split_algorithm = getattr(self, 'iterable')['split'].algorithm
        for train_index, test_index in split_algorithm.split(
                self.ingredients.x, self.ingredients.y):
            self.ingredients.x_train, self.ingredients.x_test = (
                   self.ingredients.x.iloc[train_index],
                   self.ingredients.x.iloc[test_index])
            self.ingredients.y_train, self.ingredients.y_test = (
                   self.ingredients.y.iloc[train_index],
                   self.ingredients.y.iloc[test_index])
            for step, technique in getattr(self, 'iterable').items():
                self.ingredients = technique.implement(
                       ingredients = self.ingredients,
                       plan = self)
        return self
