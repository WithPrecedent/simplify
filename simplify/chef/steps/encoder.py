"""
.. module:: encoder
:synopsis: converts categorical features to numeric ones
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.chef.composer import ChefAlgorithm as Algorithm
from simplify.chef.composer import ChefComposer as Composer
from simplify.chef.composer import ChefTechnique as Technique


@dataclass
class Encoder(Composer):
    """Transforms categorical data to numerical data."""

    name: str = 'encoder'

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self):
        self.backward = Technique(
            name = 'backward',
            module = 'category_encoders',
            algorithm = 'BackwardDifferenceEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.basen = Technique(
            name = 'basen',
            module = 'category_encoders',
            algorithm = 'BaseNEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.binary = Technique(
            name = 'binary',
            module = 'category_encoders',
            algorithm = 'BinaryEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.dummy = Technique(
            name = 'dummy',
            module = 'category_encoders',
            algorithm = 'OneHotEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.hashing = Technique(
            name = 'hashing',
            module = 'category_encoders',
            algorithm = 'HashingEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.helmert = Technique(
            name = 'helmert',
            module = 'category_encoders',
            algorithm = 'HelmertEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.james_stein = Technique(
            name = 'james_stein',
            module = 'category_encoders',
            algorithm = 'JamesSteinEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.loo = Technique(
            name = 'loo',
            module = 'category_encoders',
            algorithm = 'LeaveOneOutEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.m_estimate = Technique(
            name = 'm_estimate',
            module = 'category_encoders',
            algorithm = 'MEstimateEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.ordinal = Technique(
            name = 'ordinal',
            module = 'category_encoders',
            algorithm = 'OrdinalEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.polynomial = Technique(
            name = 'polynomial_encoder',
            module = 'category_encoders',
            algorithm = 'PolynomialEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.sum = Technique(
            name = 'sum',
            module = 'category_encoders',
            algorithm = 'SumEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.target = Technique(
            name = 'target',
            module = 'category_encoders',
            algorithm = 'TargetEncoder',
            data_parameters = {'cols': 'categoricals'})
        self.woe = Technique(
            name = 'weight_of_evidence',
            module = 'category_encoders',
            algorithm = 'WOEEncoder',
            data_parameters = {'cols': 'categoricals'})
        super().draft()
        return self
