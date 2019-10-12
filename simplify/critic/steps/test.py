"""
.. module:: tests
:synopsis: tests for model performance
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.critic.review import CriticTechnique


@dataclass
class Test(CriticTechnique):
    """Applies statistical tests to data.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """
    technique: object = None
    parameters: object = None
    name: str = 'tests'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        super().publish()
        self.options = {
            'ks_distribution': ['scipy.stats', 'ks_2samp'],
            'ks_goodness': ['scipy.stats', 'kstest'],
            'kurtosis_test': ['scipy.stats', 'kurtosistest'],
            'normal': ['scipy.stats', 'normaltest'],
            'pearson': ['scipy.stats.pearsonr']}
        return self

    def implement(self):
        self.runtime_parameters = {
            'y_true': getattr(recipe.ingredients, 'y_' + self.data_to_review),
            'y_pred': recipe.predictions}
        super().implement()
        return self