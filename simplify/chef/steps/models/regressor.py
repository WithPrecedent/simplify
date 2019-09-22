
from dataclasses import dataclass

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import (BayesianRidge, Lasso, LassoLars,
                                  LinearRegression, Ridge)
from sklearn.svm import SVR
from xgboost import XGBRegressor

from simplify.core.base import SimpleStep


@dataclass
class Regressor(SimpleStep):
    """Applies machine learning algorithms based upon user selections."""


    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'regressor'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {'adaboost' : AdaBoostRegressor,
                        'baseline_regressor' : DummyRegressor,
                        'bayes_ridge' : BayesianRidge,
                        'lasso' : Lasso,
                        'lasso_lars' : LassoLars,
                        'ols' : LinearRegression,
                        'random_forest' : RandomForestRegressor,
                        'ridge' : Ridge,
                        'svm_linear' : SVR,
                        'svm_poly' : SVR,
                        'svm_rbf' : SVR,
                        'svm_sigmoid' : SVR,
                        'xgboost' : XGBRegressor}
        self.extra_parameters = {'baseline' : {'strategy' : 'mean'},
                                 'svm_linear' : {'kernel' : 'linear',
                                                 'probability' : True},
                                  'svm_poly' : {'kernel' : 'poly',
                                                'probability' : True},
                                  'svm_rbf' : {'kernel' : 'rbf',
                                               'probability' : True},
                                  'svm_sigmoid' : {'kernel' : 'sigmoid',
                                                   'probability' : True}}
        return self

    def produce(self, ingredients):
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return self.algorithm