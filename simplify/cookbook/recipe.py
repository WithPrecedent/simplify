
from dataclasses import dataclass

from ..countertop import Countertop


@dataclass
class Recipe(Countertop):
    """Stores and creates a single recipe of steps using ingredients.

    Attributes:
        number: counter of recipe, used for file and folder naming.
        order: order for steps to be added.
        scaler: step for numerical scaling.
        splitter: step to split data into train, test, and/or validation sets.
        encoder: step to encode categorical variables.
        mixer: step for creating interactions between variables.
        cleaver: step designating subset of predictors used.
        reducer: step for feature reduction.
        model: step for machine learning technique applied.
        custom1, custom2, custom3, custom4, custom5: any custom steps added by
            user.
    """

    number : int = 0
    order : object = None
    scaler : object = None
    splitter : object = None
    encoder : object = None
    mixer : object = None
    cleaver : object = None
    sampler : object = None
    reducer : object = None
    model : object = None
    custom1 : object = None
    custom2 : object = None
    custom3 : object = None
    custom4 : object = None
    custom5 : object = None


    def __post_init__(self):
        self._set_defaults()
        return self

    def _set_defaults(self):
        """Sets order to default if none provided."""
        self.default_steps = ['scaler', 'splitter', 'encoder', 'mixer',
                              'cleaver', 'sampler', 'reducer', 'model']
        if not self.order:
            self.order = self.default_steps
        return self

    def _set_data_groups(self):
        """Copies data from ingredients to match user preferences so that the
        proper data is used for training and/or testing.
        """
        if self.data_to_use in ['train_val']:
            self.ingredients.x_test = self.ingredients.x_val
            self.ingredients.y_test = self.ingredients.y_val
        elif self.data_to_use in ['full']:
            self.ingredients.x_train = self.ingredients.x
            self.ingredients.y_train = self.ingredients.y
            self.ingredients.x_test = self.ingredients.x
            self.ingredients_y_test = self.ingredients.y
        return self

    def create(self, ingredients, data_to_use = 'train_test'):
        """Applies the Recipe methods to the passed ingredients."""
        self.ingredients = ingredients
        self.data_to_use = data_to_use
        for step in self.order:
            if step in ['model']:
                self.model.blend(ingredients = self.ingredients)
            elif step in ['reducer']:
                self.ingredients = self.reducer.blend(
                        ingredients = self.ingredients,
                        estimator = self.model.algorithm)
            else:
                step_class = getattr(self, step)
                self.ingredients = step_class.blend(
                        ingredients = self.ingredients)
                if step in ['splitter']:
                    self._set_data_groups()
        return self