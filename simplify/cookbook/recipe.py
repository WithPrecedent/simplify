
from dataclasses import dataclass
import pickle


@dataclass
class Recipe(object):
    """Stores and bakes a single recipe of siMpLify steps.

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
        customs: any custom steps added by user.
        menu: instance of Menu.
        inventory: instance of Inventory.
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
    customs : object = None
    menu : object = None
    inventory : object = None


    def __post_init__(self):
        """Initializes local attributes from menu and sets order of steps if
        one isn't passed when the class is instanced.
        """
        if self.menu:
            self.menu.localize(instance = self, sections = ['general'])
        else:
            error = 'Recipe requires an instance of Menu'
            raise AttributeError(error)
        if not self.inventory:
            error = 'Recipe requires an instance of Filer'
            raise AttributeError(error)
        # If order isn't passed, a default order is used.
        self._set_defaults()
        return self

    def _customs(self):
        """Generic custom step blender."""
        self.ingredients = self.custom.blend(ingredients = self.ingredients)
        return self

    def _encoders(self):
        """Calls appropriate encoder technique."""
        self.ingredients = self.encoder.blend(
                ingredients = self.ingredients, columns = self.ingredients.encoder_columns)
        return self

    def _mixers(self):
        """Calls appropriate mixer technique."""
        self.ingredients = self.mixer.blend(
                ingredients = self.ingredients, columns = self.ingredients.mixer_columns)
        return self

    def _models(self):
        """Calls appropriate model technique."""
        self.model.blend(ingredients = self.ingredients)
        return self

    def _samplers(self):
        """Calls appropriate sampler technique."""
        self.ingredients = self.sampler.blend(
                ingredients = self.ingredients, columns = self.ingredients.category_columns)
        return self

    def _scalers(self):
        """Calls appropriate scaler technique."""
        self.ingredients = self.scaler.blend(
                ingredients = self.ingredients, columns = self.ingredients.scaler_columns)
        return self

    def _reducers(self):
        """Calls appropriate reducer technique."""
        self.ingredients = self.reducer.blend(
                ingredients = self.ingredients, estimator = self.model.algorithm)
        return self

    def _set_defaults(self):
        """Sets order to default if none provided."""
        self.default_steps = ['scaler', 'splitter', 'encoder',
                                    'mixer', 'cleaver', 'sampler',
                                    'reducer', 'model', 'evaluator']
        if not self.order:
            self.order = self.default_steps
        return self

    def _set_data_groups(self):
        """Copies data from ingredients to match user preferences so that the proper
        data is used for training and/or testing.
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

    def _cleavers(self):
        """Calls appropriate cleaver technique."""
        self.ingredients = self.cleaver.blend(ingredients = self.ingredients)
        return self

    def _splitter(self):
        """Calls appropriate splitter technique and then sets data groups."""
        self.ingredients = self.splitter.blend(ingredients = self.ingredients)
        self._set_data_groups()
        return self

    def create(self, ingredients, data_to_use = 'train_test'):
        """Applies the Recipe methods to the passed ingredients."""
        self.ingredients = ingredients
        self.data_to_use = data_to_use
        for step in self.order:
            if step in self.default_steps:
                getattr(self, '_' + step)()
            else:
                if self.customs[step].return_codex:
                    self.codex = (
                            self.customs[step].blend(codex = self.codex))
                else:
                    self.customs[step].blend(ingredients = self.ingredients)
        return self

    def load(self, import_path):
        """Imports a single recipe from disc."""
        recipe = pickle.load(open(import_path, 'rb'))
        return recipe

    def save(self, recipe, export_path = None):
        """Exports a recipe to disc."""
        if not export_path:
            export_path = self.inventory.results_folder
        pickle.dump(recipe, open(export_path, 'wb'))
        return self