
from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from simplify.core.base import SimpleStep


@dataclass
class Classifier(SimpleStep):
    """Applies machine learning classifier algorithms based upon user
    selections.

    Args:
        technique(str): name of technique - it should always be 'gauss'
        parameters(dict): dictionary of parameters to pass to selected technique
            algorithm.
        auto_finalize(bool): whether 'finalize' method should be called when the
            class is instanced. This should generally be set to True.
        name(str): name of class for matching settings in the Idea instance and
            for labeling the columns in files exported by Critic.
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
        self.extra_parameters = {'baseline' : {'strategy' : 'most_frequent'},
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