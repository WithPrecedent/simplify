"""
.. module:: combine
:synopsis: combines data columns into new columns
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np

from simplify.core.technique import FarmerTechnique


@dataclass
class Combine(FarmerTechnique):
    """Combines features into new features.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'combiner'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init()
        return self

    def _combine_all(self, ingredients):
        ingredients.df[self.parameters['out_column']] = np.where(
                np.all(ingredients.df[self.parameters['in_columns']]),
                True, False)
        return ingredients

    def _combine_any(self, ingredients):
        ingredients.df[self.parameters['out_column']] = np.where(
                np.any(ingredients.df[self.parameters['in_columns']]),
                True, False)
        return ingredients

    def _dict(self, ingredients):
        ingredients.df[self.parameters['out_column']] = (
                ingredients.df[self.parameters['in_columns']].map(
                        self.method))
        return ingredients

    def draft(self) -> None:
        self.options = {'all': self._combine_all,
                        'any': self._combine_any,
                        'dict': self._dict}
        if isinstance(self.method, str):
            self.algorithm = self.options[self.method]
        else:
            self.algorithm = self._dict
        return self

    def publish(self, ingredients):
        self.ingredients = self.algorithm(ingredients)
        return ingredients