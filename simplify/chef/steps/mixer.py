"""
.. module:: mixer
:synopsis: combines features into new features
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.step import SimpleStep
from simplify.core.step import SimpleDesign


@dataclass
class Mixer(SimpleStep):
    """Computes new features by combining existing ones.
    
    Args: 
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
            
    """

    name: str = 'mixer'

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.options = {
        'polynomial': SimpleDesign(
            name = 'polynomial_mixer',
            module = 'sklearn.preprocessing',
            algorithm = 'PolynomialFeatures',
            default = {
                'degree': 2,
                'interaction_only': True,
                'include_bias': True}),
        'quotient': SimpleDesign(
            name = 'quotient',
            module = None,
            algorithm = 'QuotientFeatures'),
        'sum': SimpleDesign(
            name = 'sum',
            module = None,
            algorithm = 'SumFeatures'),
        'difference': SimpleDesign(
            name = 'difference',
            module = None,
            algorithm = 'DifferenceFeatures')}
        return self


# @dataclass
# class DifferenceFeatures(Algorithm):
#     """[summary]

#     Args:
#         technique (str):
#         parameters (dict):
#         space (dict):
#     """
#     technique: str
#     parameters: object
#     space: object

#     def __post_init__(self):
#         self.idea_sections = ['chef']
#         super().__post_init__()
#         return self


# @dataclass
# class QuotientFeatures(Algorithm):
#     """[summary]

#     Args:
#         technique (str):
#         parameters (dict):
#         space (dict):
#     """
#     technique: str
#     parameters: object
#     space: object

#     def __post_init__(self):
#         self.idea_sections = ['chef']
#         super().__post_init__()
#         return self


# @dataclass
# class SumFeatures(Algorithm):
#     """[summary]

#     Args:
#         technique (str):
#         parameters (dict):
#         space (dict):
#     """
#     technique: str
#     parameters: object
#     space: object

#     def __post_init__(self):
#         self.idea_sections = ['chef']
#         super().__post_init__()
#         return self