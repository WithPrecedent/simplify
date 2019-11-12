"""
.. module:: sampler
:synopsis: synthetically resamples data to different distributions
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleComposer
from simplify.core.technique import SimpleDesign


@dataclass
class Sampler(SimpleComposite):
    """Resamples data based on outcome distributions.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'sampler'

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self.options = {
            'adasyn': SimpleDesign(
                name = 'adasyn',
                module = 'imblearn.over_sampling',
                algorithm = 'ADASYN',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'cluster': SimpleDesign(
                name = 'cluster',
                module = 'imblearn.under_sampling',
                algorithm = 'ClusterCentroids',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'knn': SimpleDesign(
                name = 'knn',
                module = 'imblearn.under_sampling',
                algorithm = 'AllKNN',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'near_miss': SimpleDesign(
                name = 'near_miss',
                module = 'imblearn.under_sampling',
                algorithm = 'NearMiss',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'random_over': SimpleDesign(
                name = 'random_over',
                module = 'imblearn.over_sampling',
                algorithm = 'RandomOverSampler',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'random_under': SimpleDesign(
                name = 'random_under',
                module = 'imblearn.under_sampling',
                algorithm = 'RandomUnderSampler',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'smote': SimpleDesign(
                name = 'smote',
                module = 'imblearn.over_sampling',
                algorithm = 'SMOTE',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'smotenc': SimpleDesign(
                name = 'smotenc',
                module = 'imblearn.over_sampling',
                algorithm = 'SMOTENC',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'},
                data_dependent = {
                    'categorical_features' : 'categoricals_indices'}),
            'smoteenn': SimpleDesign(
                name = 'smoteenn',
                module = 'imblearn.combine',
                algorithm = 'SMOTEENN',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'smotetomek': SimpleDesign(
                name = 'smotetomek',
                module = 'imblearn.combine',
                algorithm = 'SMOTETomek',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'})}
        return self