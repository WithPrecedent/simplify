"""
.. module:: encoder
:synopsis: converts categorical features to numeric ones
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.step import SimpleStep
from simplify.core.step import SimpleDesign


@dataclass
class Encoder(SimpleStep):
    """Transforms categorical data to numerical data.
    
    Args: 
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
            
    """

    name: str = 'encoder'

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self.options = {
            'backward': SimpleDesign(
                name = 'backward',
                module = 'category_encoders',
                algorithm = 'BackwardDifferenceEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'basen': SimpleDesign(
                name = 'basen',
                module = 'category_encoders',
                algorithm = 'BaseNEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'binary': SimpleDesign(
                name = 'binary',
                module = 'category_encoders',
                algorithm = 'BinaryEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'dummy': SimpleDesign(
                name = 'dummy',
                module = 'category_encoders',
                algorithm = 'OneHotEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'hashing': SimpleDesign(
                name = 'hashing',
                module = 'category_encoders',
                algorithm = 'HashingEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'helmert': SimpleDesign(
                name = 'helmert',
                module = 'category_encoders',
                algorithm = 'HelmertEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'james_stein': SimpleDesign(
                name = 'james_stein',
                module = 'category_encoders',
                algorithm = 'JamesSteinEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'loo': SimpleDesign(
                name = 'loo',
                module = 'category_encoders',
                algorithm = 'LeaveOneOutEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'm_estimate': SimpleDesign(
                name = 'm_estimate',
                module = 'category_encoders',
                algorithm = 'MEstimateEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'ordinal': SimpleDesign(
                name = 'ordinal',
                module = 'category_encoders',
                algorithm = 'OrdinalEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'polynomial': SimpleDesign(
                name = 'polynomial_encoder',
                module = 'category_encoders',
                algorithm = 'PolynomialEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'sum': SimpleDesign(
                name = 'sum',
                module = 'category_encoders',
                algorithm = 'SumEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'target': SimpleDesign(
                name = 'target',
                module = 'category_encoders',
                algorithm = 'TargetEncoder',
                data_parameters = {'cols': 'categoricals'}),
            'woe': SimpleDesign(
                name = 'weight_of_evidence',
                module = 'category_encoders',
                algorithm = 'WOEEncoder',
                data_parameters = {'cols': 'categoricals'})}
        return self
