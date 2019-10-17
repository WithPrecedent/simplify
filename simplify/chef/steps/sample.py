"""
.. module:: sample
:synopsis: synthetically resamples data to different distributions
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

from simplify.core.technique import ChefTechnique
from simplify.core.decorators import numpy_shield


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'adasyn': ['imblearn.over_sampling', 'ADASYN'],
    'cluster': ['imblearn.under_sampling', 'ClusterCentroids'],
    'knn': ['imblearn.under_sampling', 'AllKNN'],
    'near_miss': ['imblearn.under_sampling', 'NearMiss'],
    'random_over': ['imblearn.over_sampling', 'RandomOverSampler'],
    'random_under': ['imblearn.under_sampling', 'RandomUnderSampler'],
    'smote': ['imblearn.over_sampling', 'SMOTE'],
    'smotenc': ['imblearn.over_sampling', 'SMOTENC'],
    'smoteenn':  ['imblearn.combine', 'SMOTEENN'],
    'smotetomek': ['imblearn.combine', 'SMOTETomek']}


@dataclass
class Sample(ChefTechnique):
    """Synthetically resamples data according to selected algorithm.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'sample'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self


    def _recheck_parameters(self, ingredients, columns = None):
        if self.technique in ['smotenc']:
            if columns:
                cat_features = self._get_indices(ingredients.x, columns)
                self.parameters.update({'categorical_features': cat_features})
            else:
                cat_features = self._get_indices(ingredients.x,
                                                 ingredients.categoricals)
        return self


    def draft(self):
        super().draft()
        self.default_parameters = {'sampling_strategy': 'auto'}
        return self

    def publish(self):
        self.runtime_parameters = {'random_state': self.seed}
        super().publish()
        return self

    @numpy_shield
    def implement(self, ingredients, plan = None, columns = None):
        self._recheck_parameters(ingredients.x, columns)
        if plan.data_to_use in ['full']:
            resampled_x, resampled_y = self.algorithm.fit_resample(
                    ingredients.x, ingredients.y)
            ingredients.x, ingredients.y = self._get_column_names(
                    resampled_x, resampled_y)
        else:
            resampled_x, resampled_y = self.algorithm.fit_resample(
                    ingredients.x_train, ingredients.y_train)
        return ingredients

    def fit(self, x, y, columns = None):
        self._recheck_parameters(x, columns)
        return self

    def fit_transform(self, x, y):
        x = self.transform(x, y)
        return x

    def transform(self, x, y):
        x = self.algorithm.fit_resample(x, y)
        return x