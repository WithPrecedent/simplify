"""
.. module:: regress
:synopsis: machine learning regression algorithms
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

from simplify.core.technique import ChefTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'adaboost': ['sklearn.ensemble', 'AdaBoostRegressor'],
    'baseline_regressor': ['sklearn.dummy', 'DummyRegressor'],
    'bayes_ridge': ['sklearn.linear_model', 'BayesianRidge'],
    'lasso': ['sklearn.linear_model', 'Lasso'],
    'lasso_lars': ['sklearn.linear_model', 'LassoLars'],
    'ols': ['sklearn.linear_model', 'LinearRegression'],
    'random_forest': ['sklearn.ensemble', 'RandomForestRegressor'],
    'ridge': ['sklearn.linear_model', 'Ridge'],
    'svm_linear': ['sklearn.svm', 'SVR'],
    'svm_poly': ['sklearn.svm', 'SVR'],
    'svm_rbf': ['sklearn.svm', 'SVR'],
    'svm_sigmoid': ['sklearn.svm', 'SVR'],
    'xgboost': ['xgboost', 'XGBRegressor']}


@dataclass
class Regress(ChefTechnique):
    """Applies machine learning algorithms based upon user selections.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    auto_publish: bool = True
    name: str = 'regressor'
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)
    
    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Private Methods """

    def _get_conditional_options(self):
        if self.gpu:
            self.options.update({
                'lasso': ['cuml', 'Lasso'],
                'ols': ['cuml', 'LinearRegression'],
                'ridge': ['cuml', 'RidgeRegression']})
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.extra_parameters = {
            'baseline': {'strategy': 'mean'},
            'svm_linear': {'kernel': 'linear',
                            'probability': True},
            'svm_poly': {'kernel': 'poly',
                        'probability': True},
            'svm_rbf': {'kernel': 'rbf',
                        'probability': True},
            'svm_sigmoid': {'kernel': 'sigmoid',
                            'probability': True}}
        self._get_conditional_options()
        return self

    def implement(self, ingredients):
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return self.algorithm