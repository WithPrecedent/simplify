"""
.. module:: model
:synopsis: Applies machine learning and statistical models to data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from scipy.stats import randint, uniform

from simplify.core.step import SimpleStep


@dataclass
class Model(SimpleStep):
    """Applies machine learning algorithms based upon user selections.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'model'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['cookbook']
        super().__post_init__()
        return self

    """ Private Methods """

    def _datatype_in_list(self, test_list, data_type):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, data_type) for i in test_list)

    def _publish_options_gpu(self):
        self.classifier_algorithms.update({
                'forest_inference': ['cuml', 'ForestInference'],
                'random_forest': ['cuml', 'RandomForestClassifier'],
                'logit': ['cuml', 'LogisticRegression']})
        self.clusterer_algorithms.update({
                'dbscan': ['cuml', 'DBScan'],
                'kmeans': ['cuml', 'KMeans']})
        self.regressor_algorithms.update({
                'lasso': ['cuml', 'Lasso'],
                'ols': ['cuml', 'LinearRegression'],
                'ridge': ['cuml', 'RidgeRegression']})
        return self

    def _publish_search_parameters(self):
        self.search_parameters = self.idea['search_parameters']
        self.search_parameters.update({'estimator': self.algorithm.algorithm,
                                       'param_distributions': self.space,
                                       'random_state': self.seed})
        return self

    def _get_parameters_conditional(self, technique, parameters):
        if (technique in ['xgboost']
                and not 'scale_pos_weight' in parameters
                and hasattr(self, 'scale_pos_weight')):
            self.parameters.update(
                    {'scale_pos_weight': self.scale_pos_weight})
            if self.gpu:
                self.parameters.update({'tree_method': 'gpu_exact'})
        return self

    def _parse_parameters(self):
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
                    self.space.update({param: uniform(values[0], values[1])})
                elif self._datatype_in_list(values, int):
                    self.space.update({param: randint(values[0], values[1])})
            else:
                new_parameters.update({param: values})
        self.parameters = new_parameters
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.options = {
                'classify': ['simplify.chef.steps.techniques.classify',
                             'Classify'],
                'cluster': ['simplify.chef.steps.techniques.cluster',
                            'Cluster'],
                'regress': ['simplify.chef.steps.techniques.regress',
                            'Regress'],
                'search': ['simplify.chef.steps.techniques.search', 'Search']}
        self.runtime_parameters = {'random_state': self.seed}
        self.custom_options = ['classifier', 'clusterer', 'regressor']
        return self

    def publish(self):
        if self.technique != 'none':
            self._publish_parameters()
            self._parse_parameters()
            self.algorithm = self.options[self.model_type](
                technique = self.technique,
                parameters = self.parameters)
            if self.hyperparameter_search:
                self._publish_search_parameters()
                self.search_algorithm = self.options['search'](
                        technique = self.search_technique,
                        parameters = self.search_parameters)

        return self

    def implement(self, ingredients, plan = None):
        """Applies model from recipe to ingredients data."""
        if self.technique != 'none':
            if self.hyperparameter_search:
                self.algorithm = self.search_algorithm.implement(
                        ingredients = ingredients)
            else:
                self.algorithm = self.algorithm.implement(
                        ingredients = ingredients)
        return ingredients

    """ Scikit-Learn Compatibility Methods """

    def fit_transform(self, x, y = None):
        error = 'fit_transform is not implemented for machine learning models'
        raise NotImplementedError(error)

    def transform(self, x, y = None):
        error = 'transform is not implemented for machine learning models'
        raise NotImplementedError(error)