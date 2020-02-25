"""
.. module:: combine
:synopsis: combines data columns into new columns
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses import dataclass

import numpy as np

from simplify.core.definitionsetter import WranglerTechnique


@dataclass
class Combine(WranglerTechnique):
    """Combines features into new features.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'combiner'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init()
        return self

    def _combine_all(self, dataset):
        dataset.df[self.parameters['out_column']] = np.where(
                np.all(dataset.df[self.parameters['in_columns']]),
                True, False)
        return dataset

    def _combine_any(self, dataset):
        dataset.df[self.parameters['out_column']] = np.where(
                np.any(dataset.df[self.parameters['in_columns']]),
                True, False)
        return dataset

    def _dict(self, dataset):
        dataset.df[self.parameters['out_column']] = (
                dataset.df[self.parameters['in_columns']].map(
                        self.method))
        return dataset

    def draft(self) -> None:
        self._options = Repository(contents = {'all': self._combine_all,
                        'any': self._combine_any,
                        'dict': self._dict}
        if isinstance(self.method, str):
            self.algorithm = self.workers[self.method]
        else:
            self.algorithm = self._dict
        return self

    def publish(self, dataset):
        self.data = self.algorithm(dataset)
        return dataset