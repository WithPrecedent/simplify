"""
.. module:: encoder
:synopsis: converts categorical features to numeric ones
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleComposer
from simplify.core.technique import SimpleDesign


@dataclass
class Encoder(SimpleComposer):
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
                data_dependent = {'cols': 'categoricals'}),
            'basen': SimpleDesign(
                name = 'basen',
                module = 'category_encoders',
                algorithm = 'BaseNEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'binary': SimpleDesign(
                name = 'binary',
                module = 'category_encoders',
                algorithm = 'BinaryEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'dummy': SimpleDesign(
                name = 'dummy',
                module = 'category_encoders',
                algorithm = 'OneHotEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'hashing': SimpleDesign(
                name = 'hashing',
                module = 'category_encoders',
                algorithm = 'HashingEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'helmert': SimpleDesign(
                name = 'helmert',
                module = 'category_encoders',
                algorithm = 'HelmertEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'james_stein': SimpleDesign(
                name = 'james_stein',
                module = 'category_encoders',
                algorithm = 'JamesSteinEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'loo': SimpleDesign(
                name = 'loo',
                module = 'category_encoders',
                algorithm = 'LeaveOneOutEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'm_estimate': SimpleDesign(
                name = 'm_estimate',
                module = 'category_encoders',
                algorithm = 'MEstimateEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'ordinal': SimpleDesign(
                name = 'ordinal',
                module = 'category_encoders',
                algorithm = 'OrdinalEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'polynomial': SimpleDesign(
                name = 'polynomial_encoder',
                module = 'category_encoders',
                algorithm = 'PolynomialEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'sum': SimpleDesign(
                name = 'sum',
                module = 'category_encoders',
                algorithm = 'SumEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'target': SimpleDesign(
                name = 'target',
                module = 'category_encoders',
                algorithm = 'TargetEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'woe': SimpleDesign(
                name = 'weight_of_evidence',
                module = 'category_encoders',
                algorithm = 'WOEEncoder',
                data_dependent = {'cols': 'categoricals'})}
        return self
