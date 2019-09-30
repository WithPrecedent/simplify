"""
.. module:: regress
:synopsis: machine learning regression algorithms
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.base import SimpleTechnique


@dataclass
class Regress(SimpleTechnique):
    """Applies machine learning algorithms based upon user selections.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    auto_finalize: bool = True
    name: str = 'regressor'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.options = {
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
        self.extra_parameters = {'baseline': {'strategy': 'mean'},
                                 'svm_linear': {'kernel': 'linear',
                                                 'probability': True},
                                  'svm_poly': {'kernel': 'poly',
                                                'probability': True},
                                  'svm_rbf': {'kernel': 'rbf',
                                               'probability': True},
                                  'svm_sigmoid': {'kernel': 'sigmoid',
                                                   'probability': True}}
        return self

    def produce(self, ingredients):
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return self.algorithm