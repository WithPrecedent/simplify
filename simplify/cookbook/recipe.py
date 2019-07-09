
from dataclasses import dataclass

@dataclass
class Recipe(object):
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

    def __post_init__(self):
        return self

    def prepare(self):
        return self

    def start(self, ingredients, data_to_use = 'train_test'):
        """Applies the Recipe methods to the passed ingredients."""
        self.ingredients = ingredients
        self.ingredients._remap_dataframes(data_to_use = data_to_use)
        self.data_to_use = data_to_use
        if 'val' in data_to_use:
            self.val_set = True
        else:
            self.val_set = False
        for step in self.order:
            self.ingredients = getattr(self, step).start(
                    ingredients = self.ingredients, recipe = self)
        return self