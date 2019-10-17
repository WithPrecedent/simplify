"""
.. module:: classify
:synopsis: machine learning algorithms for classification problems
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


@dataclass
class Classify(ChefTechnique):
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
    name: str = 'classify'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Private Methods """

    def _get_conditional_options(self):
        if self.idea['general']['gpu']:
            self.options.update({
                'forest_inference': ['cuml', 'ForestInference'],
                'random_forest': ['cuml', 'RandomForestClassifier'],
                'logit': ['cuml', 'LogisticRegression']})
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.extra_parameters = {
            'baseline': {'strategy': 'most_frequent'},
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