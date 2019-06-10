
from dataclasses import dataclass
import pickle

@dataclass
class Recipe(object):
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
        custom: any custom ingredient added by user.
        model: ingredient for machine learning technique applied.
        evaluator: ingredient designating metrics to be used to assess model
            performance.
        plotter: ingredient of visualizations of model and evaluator.
        settings: instance of Settings.
        filer: instance of Filer.

    settings and filer are automatically injected into Recipe if Cookbook is
    used. If Recipe is used independent of the Cookbook, settings and filer
    must be passed."""

    number : int = 0
    order : object = None
    scaler : object = None
    splitter : object = None
    encoder : object = None
    interactor : object = None
    splicer : object = None
    sampler : object = None
    selector : object = None
    custom : object = None
    model : object = None
    evaluator : object = None
    plotter : object = None
    settings : object = None
    filer : object = None

    def __post_init__(self):
        """Initializes local attributes from settings and sets order of
        ingredients if one isn't passed when the class is instanced.
        """
        if self.settings:
            self.settings.localize(instance = self, sections = ['general'])
        else:
            error = 'Recipe requires an instance of Settings'
            raise AttributeError(error)
        if not self.filer:
            error = 'Recipe requires an instance of Filer'
            raise AttributeError(error)
        self._set_order()
        return self

    def _set_order(self):
        """Sets order to default if none provided."""
        default_order = ['scaler', 'splitter', 'encoder', 'interactor',
                         'splicer', 'sampler', 'selector', 'custom', 'model',
                         'evaluator', 'explainer', 'plotter']
        if not self.order:
            self.order = default_order
        return self

    def _set_data_group(self):
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

    def _scalers(self):
        """Calls appropriate scaler technique."""
        self.codex = self.scaler.mix(
                codex = self.codex, columns = self.codex.scaler_columns)
        return self

    def _splitter(self):
        """Calls appropriate splitter technique and then sets data groups."""
        self.codex = self.splitter.mix(codex = self.codex)
        self._set_data_group()
        return self

    def _encoders(self):
        """Calls appropriate encoder technique."""
        self.codex = self.encoder.mix(
                codex = self.codex, columns = self.codex.encoder_columns)
        return self

    def _interactors(self):
        """Calls appropriate interactor technique."""
        self.codex = self.interactor.mix(
                codex = self.codex, columns = self.codex.interactor_columns)
        return self

    def _splicers(self):
        """Calls appropriate splicer technique."""
        self.codex = self.splicer.mix(codex = self.codex)
        return self

    def _samplers(self):
        """Calls appropriate sampler technique."""
        self.codex = self.sampler.mix(
                codex = self.codex, columns = self.codex.category_columns)
        return self

    def _customs(self):
        """Calls appropriate custom technique."""
        self.codex = self.custom.mix(codex = self.codex)
        return self

    def _selectors(self):
        """Calls appropriate selector technique."""
        self.codex = self.selector.mix(
                codex = self.codex, estimator = self.model.algorithm)
        return self

    def _models(self):
        """Calls appropriate model technique."""
        self.model.mix(codex = self.codex)
        return self

    def _evaluator(self):
        """Calls appropriate evalutor techniques."""
        self.evaluator.mix(recipe = self, data_to_use = self.data_to_use)
        return self

    def _plotter(self):
        """Calls appropriate plotting techniques."""
        self.plotter.mix(recipe = self)
        return self

    def load(self, import_path = None):
        """Imports a single recipe from disc."""
        recipe = pickle.load(open(import_path, 'rb'))
        return recipe

    def save(self, recipe, export_path = None):
        """Exports a recipe to disc."""
        if not export_path:
            export_path = self.filer.results_folder
        pickle.dump(recipe, open(export_path, 'wb'))
        return self

    def bake(self, codex, data_to_use = 'train_test'):
        """Applies the Recipe methods to the passed codex."""
        self.codex = codex
        self.data_to_use = data_to_use
        for step in self.order:
            getattr(self, '_' + step)()
        return self