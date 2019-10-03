"""
.. module:: classify
:synopsis: machine learning algorithms for classification problems
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleTechnique


@dataclass
class Classify(SimpleTechnique):
    """Applies machine learning classifier algorithms based upon user
    selections.

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
    name: str = 'classifier'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.options = {
                'adaboost': ['sklearn.ensemble', 'AdaBoostClassifier'],
                'baseline_classifier': ['sklearn.dummy', 'DummyClassifier'],
                'logit': ['sklearn.linear_model', 'LogisticRegression'],
                'random_forest': ['sklearn.ensemble',
                                   'RandomForestClassifier'],
                'svm_linear': ['sklearn.svm', 'SVC'],
                'svm_poly': ['sklearn.svm', 'SVC'],
                'svm_rbf': ['sklearn.svm', 'SVC'],
                'svm_sigmoid': ['sklearn.svm', 'SVC'],
                'xgboost': ['xgboost', 'XGBClassifier']}
        self.extra_parameters = {'baseline': {'strategy': 'most_frequent'},
                                 'svm_linear': {'kernel': 'linear',
                                                 'probability': True},
                                 'svm_poly': {'kernel': 'poly',
                                               'probability': True},
                                 'svm_rbf': {'kernel': 'rbf',
                                              'probability': True},
                                 'svm_sigmoid': {'kernel': 'sigmoid',
                                                  'probability': True}}
        return self

    def publish(self):
        if self.technique in self.extra_parameters:
            self.parameters.update(self.extra_parameters[self.technique])
        if self.technique != ['none']:
            self.algorithm = self.options[self.technique](**self.parameters)
        else:
            self.algorithm = None
        return self

    def implement(self, ingredients):
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return self.algorithm