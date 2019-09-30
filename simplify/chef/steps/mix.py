"""
.. module:: mix
:synopsis: combines features into new features
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.base import SimpleStep
from simplify.core.decorators import numpy_shield


@dataclass
class Mix(SimpleStep):
    """Computes new features using different algorithms selected.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: str = ''
    parameters: object = None
    name: str = 'mixer'
    auto_finalize: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.options = {
                'polynomial': ['category_encoders', 'PolynomialEncoder'],
                'quotient': ['simplify.chef.steps.techniques.mixers',
                             'QuotientFeatures'],
                'sum': ['simplify.chef.steps.techniques.mixers',
                        'SumFeatures'],
                'difference': ['simplify.chef.steps.techniques.mixers',
                               'DifferenceFeatures']}
        return self

    def quotient_features(self):
        pass
        return self

    def sum_features(self):
        pass
        return self

    def difference_features(self):
        pass
        return self

    def finalize(self):
        pass
        return self

    @numpy_shield
    def produce(self, ingredients, plan = None, columns = None):
        if not columns:
            columns = ingredients.encoders
        if columns:
            self.runtime_parameters = {'cols': columns}
        super().finalize()
        self.algorithm.fit(ingredients.x, ingredients.y)
        self.algorithm.transform(
                ingredients.x_train).reset_index(drop = True)
        self.algorithm.transform(
                ingredients.x_test).reset_index(drop = True)
        return ingredients