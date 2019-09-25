
from dataclasses import dataclass

from scipy.stats import randint, uniform

from simplify.chef.steps.models.classifier import Classifier
from simplify.chef.steps.models.clusterer import Clusterer
from simplify.chef.steps.models.regressor import Regressor
from simplify.core.base import SimpleStep


@dataclass
class Model(SimpleStep):
    """Applies machine learning algorithms based upon user selections.

    Args:
        technique(str): name of technique - it should always be 'gauss'
        parameters(dict): dictionary of parameters to pass to selected technique
            algorithm.
        auto_finalize(bool): whether 'finalize' method should be called when the
            class is instanced. This should generally be set to True.
        name(str): name of class for matching settings in the Idea instance and
            for labeling the columns in files exported by Critic.
    """
    techniques : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'model'

    def __post_init__(self):
        self.idea_sections = ['cookbook']
        super().__post_init__()
        return self

    def _check_specific_parameters(self):
        if ('xgboost' in self.model_parameters
                and not 'scale_pos_weight' in self.model_parameters['xgboost']
                and hasattr(self, 'scale_pos_weight')):
            self.parameters['xgboost'].update(
                    {'scale_pos_weight' : self.scale_pos_weight})
        if 'xgboost' in self.model_parameters and self.gpu:
            self.runtime_parameters.update(
                    {'xgboost' : {'tree_method' : 'gpu_exact'}})
        return self

    def _datatype_in_list(self, test_list, data_type):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, data_type) for i in test_list)


    # def finalize(self):
    #     self._edit_specific_parameters()
    #     if self.gpu:
    #        self._gpu_parameters()
    #     self._parse_parameters()
    #     self.estimator = self.options[self.technique]
    #     if self.hyperparameter_search:
    #         parameters = {'estimator' : self.estimator,
    #                       'space' : self.space}
    #         self.searcher = Search(technique = self.search_technique,
    #                                parameters = parameters )
    #     self.algorithm = self.estimator(**self.parameters)
    #     return self

    # def produce(self, ingredients):
    #     if self.hyperparameter_search:
    #         self.searcher.produce(ingredients = ingredients)
    #         self.algorithm = self.searcher.best_estimator
    #     else:
    #         self.algorithm.fit(ingredients.x_train, ingredients.y_train)
    #     return ingredients

    def _parse_search_parameters(self):
        """Parses parameters to determine if the user has created ranges of
        parameters. If so, parameters are divided between those to be searched
        and those that are fixed. If any parameters include a range of values,
        hyperparameter_search is set to True. Fixed parameters are stored in
        the parameters attribute. Ranges to be searched are stored in the
        space attribute.
        """
        self.hyperparameter_search = False
        self.space = {}
        new_parameters = {}
        for param, values in self.parameters.items():
            if isinstance(values, list):
                self.hyperparameter_search = True
                if self._datatype_in_list(values, float):
                    self.space.update({param : uniform(values[0], values[1])})
                elif self._datatype_in_list(values, int):
                    self.space.update({param : randint(values[0], values[1])})
            else:
                new_parameters.update({param : values})
        self.parameters = new_parameters
        return self

    def draft(self):
        self.options = {'classifier' : Classifier,
                        'clusterer' : Clusterer,
                        'regressor' : Regressor}
        self.runtime_parameters = {'random_state' : self.seed}
        self.check_nests.append('model_parameters')
        return self

    def finalize(self):
        """Adds parameters to machine learning algorithm."""
        if self.technique != 'none':
            if not hasattr(self, 'parameters') or not self.parameters:
                self.model_parameters = self.idea[self.technique]
            self._nestify_parameters()
            self._check_specific_parameters()
            self._parse_search_parameters()
            self._finalize_parameters()
            self.algorithm = self.options[self.model_type](
                    technique = self.technique,
                    parameters = self.parameters)
            self.algorithm.finalize()
        return self

    def fit_transform(self, x, y = None):
        error = 'fit_transform is not implemented for machine learning models'
        raise NotImplementedError(error)

    def produce(self, ingredients, plan = None):
        """Applies model from recipe to ingredients data."""
        if self.technique != 'none':
            if self.verbose:
                print('Applying', self.technique, 'model')
            self.algorithm = self.algorithm.produce(ingredients = ingredients)
        return self

    def transform(self, x, y = None):
        error = 'transform is not implemented for machine learning models'
        raise NotImplementedError(error)