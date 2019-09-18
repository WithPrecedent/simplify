
from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from simplify.chef.steps.models.algorithm import Algorithm


@dataclass
class Classifier(Algorithm):
    """Applies machine learning classifier algorithms based upon user
    selections.
    """
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'classifier'

    def __post_init__(self):
        
        super().__post_init__()
        return self

    def draft(self):
        self.checks = ['idea']
        self.options = {'adaboost' : AdaBoostClassifier,
                        'baseline_classifier' : DummyClassifier,
                        'logit' : LogisticRegression,
                        'random_forest' : RandomForestClassifier,
                        'svm_linear' : SVC,
                        'svm_poly' : SVC,
                        'svm_rbf' : SVC,
                        'svm_sigmoid' : SVC,
                        'xgboost' : XGBClassifier}
        self.model_parameters = {'baseline' : {'strategy' : 'most_frequent'},
                                 'svm_linear' : {'kernel' : 'linear',
                                                 'probability' : True},
                                 'svm_poly' : {'kernel' : 'poly',
                                               'probability' : True},
                                 'svm_rbf' : {'kernel' : 'rbf',
                                              'probability' : True},
                                 'svm_sigmoid' : {'kernel' : 'sigmoid',
                                                  'probability' : True}}
        return self
