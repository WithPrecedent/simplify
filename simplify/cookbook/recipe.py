
from dataclasses import dataclass
import pickle

from .steps.custom import Custom
from .steps.step import Step


@dataclass
class Recipe(Step):
    """Stores and bakes a single recipe of siMpLify ingredients.

    Attributes:
        number: counter of recipe, used for file and folder naming.
        order: order for ingredients to be added.
        scaler: ingredient for numerical scaling.
        splitter: ingredient to split data into train, test, and/or validation
            sets.
        encoder: ingredient to encode categorical variables.
        interactor: ingredient for creating interactions between variables.
        splicer: ingredient designating subset of predictors used.
        selector: ingredient for feature reduction.
        model: ingredient for machine learning technique applied.
        customs: any custom ingredients added by user.
        menu: instance of Settings.
        pantry: instance of Filer.

    menu and pantry are automatically injected into Recipe if Cookbook is
    used. If Recipe is used independent of the Cookbook, menu and pantry
    must be passed.
    """

    number : int = 0
    order : object = None
    scaler : object = None
    splitter : object = None
    encoder : object = None
    interactor : object = None
    splicer : object = None
    sampler : object = None
    selector : object = None
    model : object = None
    customs : object = None
    menu : object = None
    pantry : object = None


    def __post_init__(self):
        """Initializes local attributes from menu and sets order of
        ingredients if one isn't passed when the class is instanced.
        """
        if self.menu:
            self.menu.localize(instance = self, sections = ['general'])
        else:
            error = 'Recipe requires an instance of Settings'
            raise AttributeError(error)
        if not self.pantry:
            error = 'Recipe requires an instance of Filer'
            raise AttributeError(error)
        # If order isn't passed, a default order is used.
        self._set_defaults()
        return self

    def _customs(self):
        """Generic custom ingredient blender."""
        self.codex = self.custom.blend(codex = self.codex)
        return self

    def _encoders(self):
        """Calls appropriate encoder technique."""
        self.codex = self.encoder.blend(
                codex = self.codex, columns = self.codex.encoder_columns)
        return self

    def _interactors(self):
        """Calls appropriate interactor technique."""
        self.codex = self.interactor.blend(
                codex = self.codex, columns = self.codex.interactor_columns)
        return self

    def _models(self):
        """Calls appropriate model technique."""
        self.model.blend(codex = self.codex)
        return self

    def _samplers(self):
        """Calls appropriate sampler technique."""
        self.codex = self.sampler.blend(
                codex = self.codex, columns = self.codex.category_columns)
        return self

    def _scalers(self):
        """Calls appropriate scaler technique."""
        self.codex = self.scaler.blend(
                codex = self.codex, columns = self.codex.scaler_columns)
        return self

    def _selectors(self):
        """Calls appropriate selector technique."""
        self.codex = self.selector.blend(
                codex = self.codex, estimator = self.model.algorithm)
        return self

    def _set_defaults(self):
        """Sets order to default if none provided."""
        self.default_ingredients = ['scaler', 'splitter', 'encoder',
                                    'interactor', 'splicer', 'sampler',
                                    'selector', 'model', 'evaluator']
        if not self.order:
            self.order = self.default_ingredients
        return self

    def _set_data_groups(self):
        """Copies data from codex to match user preferences so that the proper
        data is used for training and/or testing.
        """
        if self.data_to_use in ['train_val']:
            self.codex.x_test = self.codex.x_val
            self.codex.y_test = self.codex.y_val
        elif self.data_to_use in ['full']:
            self.codex.x_train = self.codex.x
            self.codex.y_train = self.codex.y
            self.codex.x_test = self.codex.x
            self.codex_y_test = self.codex.y
        return self

    def _splicers(self):
        """Calls appropriate splicer technique."""
        self.codex = self.splicer.blend(codex = self.codex)
        return self

    def _splitter(self):
        """Calls appropriate splitter technique and then sets data groups."""
        self.codex = self.splitter.blend(codex = self.codex)
        self._set_data_groups()
        return self

    def add_technique(self, name, parameters, func, runtime_parameters = None):
        setattr(self, name, Custom(technique = name,
                                   parameters = parameters,
                                   method = func,
                                   runtime_parameters = runtime_parameters))
        setattr(self, '_' + step, self._customs)
        return self

    def create(self, codex, data_to_use = 'train_test'):
        """Applies the Recipe methods to the passed codex."""
        self.codex = codex
        self.data_to_use = data_to_use
        for ingredient in self.order:
            if ingredient in self.default_ingredients:
                getattr(self, '_' + ingredient)()
            else:
                if self.customs[ingredient].return_codex:
                    self.codex = (
                            self.customs[ingredient].blend(codex = self.codex))
                else:
                    self.customs[ingredient].blend(codex = self.codex)
        return self

    def load(self, import_path):
        """Imports a single recipe from disc."""
        recipe = pickle.load(open(import_path, 'rb'))
        return recipe

    def save(self, recipe, export_path = None):
        """Exports a recipe to disc."""
        if not export_path:
            export_path = self.pantry.results_folder
        pickle.dump(recipe, open(export_path, 'wb'))
        return self