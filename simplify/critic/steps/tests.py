"""
.. module:: tests
:synopsis: tests for model performance
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleTechnique


@dataclass
class Tests(SimpleTechnique):

    recipe : object = None
    technique: object = None
    parameters: object = None
    name: str = 'tests'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['critic']
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
    
    def publish(self):
        self.runtime_parameters = {
            'y_true': self.recipe.ingredients.y_test,
            'y_pred': self.recipe.predictions}
        super().publish()
        return self