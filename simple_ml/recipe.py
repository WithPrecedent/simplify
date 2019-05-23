"""
The Recipe class, which contains a single recipe of methods to be applied in a
machine learning experiment.
"""
from dataclasses import dataclass
import pickle

@dataclass
class Recipe(object):
    """
    Class containing single recipe of methods.
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
    custom : object = None
    model : object = None
    evaluator : object = None
    plotter : object = None

    def __post_init__(self):
        self.settings.localize(instance = self, sections = ['general'])
        self._set_order()
        return self

    def _set_order(self):
        default_order = ['scaler', 'splitter', 'encoder', 'interactor',
                         'splicer', 'sampler', 'selector', 'custom', 'model',
                         'evaluator', 'explainer', 'plotter']
        if not self.order:
            self.order = default_order
        return self

    def _set_data_group(self):
        if self.data_to_use in ['train_val']:
            self.data.x_test = self.data.x_val
            self.data.y_test = self.data.y_val
        elif self.data_to_use in ['full']:
            self.data.x_train = self.data.x
            self.data.y_train = self.data.y
            self.data.x_test = self.data.x
            self.data_y_test = self.data.y
        return self

    def _scalers(self):
        self.data = self.scaler.mix(data = self.data,
                                    columns = self.data.scaler_columns)
        return self

    def _splitter(self):
        self.data = self.splitter.mix(data = self.data)
        self._set_data_group()
        return self

    def _encoders(self):
        self.data = self.encoder.mix(data = self.data,
                                     columns = self.data.encoder_columns)
        return self

    def _interactors(self):
        self.data = self.interactor.mix(data = self.data,
                                        columns = self.data.interactor_columns)
        return self

    def _splicers(self):
        self.data = self.splicer.mix(data = self.data)
        return self

    def _samplers(self):
        self.data = self.sampler.mix(data = self.data,
                                     columns = self.data.category_columns)
        return self

    def _customs(self):
        self.data = self.custom.mix(data = self.data,
                                    runtime_params = self.runtime_params)
        return self

    def _selectors(self):
        self.data = self.selector.mix(data = self.data,
                                      estimator = self.model.algorithm)
        return self

    def _models(self):
        self.model.mix(data = self.data)
        return self

    def _evaluator(self):
        self.evaluator.mix(recipe = self,
                           data_to_use = self.data_to_use)
        return self

    def _plotter(self):
        self.plotter.mix(recipe = self,
                         evaluator = self.evaluator)
        return self

    def load(self, import_path = None):
        """
        Imports a single recipe from disc.
        """
        recipe = pickle.load(open(import_path, 'rb'))
        return recipe

    def save(self, recipe, export_path = None):
        """
        Exports a recipe to disc.
        """
        if not export_path:
            export_path = self.filer.results_folder
#        pickle.dump(recipe, open(export_path, 'wb'))
        return self

    def bake(self, data, data_to_use = 'train_test', runtime_params = None):
        """
        Applies the Recipe methods to the passed data.
        """
        self.data = data
        self.data_to_use = data_to_use
        self.runtime_params = runtime_params
        for step in self.order:
            step_name = '_' + step
            method = getattr(self, step_name)
            method()
        return self

    def apply(self, data, use_full_set = False, use_val_set = False):
        """
        Applies the Recipe methods using a more generic method name for those
        who prefer less cooking-oriented terminology.
        """
        self.bake(data, use_full_set, use_val_set)
        return self