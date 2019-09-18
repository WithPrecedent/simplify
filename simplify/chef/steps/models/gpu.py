
from dataclasses import dataclass

from cuml import (LinearRegression, LogisticRegression, RidgeRegression,
                  Lasso)
from cuml import RandomForestClassifier
from cuml import ForestInference
from cuml import KMeans, DBScan

from xgboost import XGBClassifier

from simplify.chef.steps.models.algorithm import Algorithm


@dataclass
class GPU(Algorithm):
    """Applies machine learning classifier algorithms based upon user
    selections using GPU.
    """
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'gpu'

    def __post_init__(self):

        super().__post_init__()
        return self

    def draft(self):
        self.checks = ['idea']
        self.options = {}
        self.model_parameters = {}
        return self
