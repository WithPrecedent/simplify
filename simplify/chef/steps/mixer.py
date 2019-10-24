"""
.. module:: mixer
:synopsis: combines features into new features
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.chef.composer import ChefAlgorithm as Algorithm
from simplify.chef.composer import ChefComposer as Composer
from simplify.chef.composer import ChefTechnique as Technique


@dataclass
class Mixer(Composer):
    """Computes new features by combining existing ones.
    """

    name: str = 'mixer'

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self):
        self.polynomial = Technique(
            name = 'polynomial_mixer',
            module = 'sklearn.preprocessing',
            algorithm = 'PolynomialFeatures',
            defaults = {
                'degree': 2,
                'interaction_only': True,
                'include_bias': True})
        self.quotient = Technique(
            name = 'quotient',
            module = None,
            algorithm = 'QuotientFeatures')
        self.sum = Technique(
            name = 'sum',
            module = None,
            algorithm = 'SumFeatures')
        self.difference = Technique(
            name = 'difference',
            module = None,
            algorithm = 'DifferenceFeatures')
        super().draft()
        return self


@dataclass
class DifferenceFeatures(Algorithm):
    """[summary]

    Args:
        technique (str):
        parameters (dict):
        space (dict):
    """
    technique: str
    parameters: object
    space: object

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self


@dataclass
class QuotientFeatures(Algorithm):
    """[summary]

    Args:
        technique (str):
        parameters (dict):
        space (dict):
    """
    technique: str
    parameters: object
    space: object

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self


@dataclass
class SumFeatures(Algorithm):
    """[summary]

    Args:
        technique (str):
        parameters (dict):
        space (dict):
    """
    technique: str
    parameters: object
    space: object

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self