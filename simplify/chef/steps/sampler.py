"""
.. module:: sampler
:synopsis: synthetically resamples data to different distributions
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.chef.composer import ChefAlgorithm as Algorithm
from simplify.chef.composer import ChefComposer as Composer
from simplify.chef.composer import ChefTechnique as Technique


@dataclass
class Sampler(Composer):
    """Resamples data based on outcome distributions.
    """

    name: str = 'sampler'

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self):
        self.adasyn = Technique(
            name = 'adasyn',
            module = 'imblearn.over_sampling',
            algorithm = 'ADASYN',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        self.cluster = Technique(
            name = 'cluster',
            module = 'imblearn.under_sampling',
            algorithm = 'ClusterCentroids',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        self.knn = Technique(
            name = 'knn',
            module = 'imblearn.under_sampling',
            algorithm = 'AllKNN',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        self.near_miss = Technique(
            name = 'near_miss',
            module = 'imblearn.under_sampling',
            algorithm = 'NearMiss',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        self.random_over = Technique(
            name = 'random_over',
            module = 'imblearn.over_sampling',
            algorithm = 'RandomOverSampler',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        self.random_under = Technique(
            name = 'random_under',
            module = 'imblearn.under_sampling',
            algorithm = 'RandomUnderSampler',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        self.smote = Technique(
            name = 'smote',
            module = 'imblearn.over_sampling',
            algorithm = 'SMOTE',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        self.smotenc = Technique(
            name = 'smotenc',
            module = 'imblearn.over_sampling',
            algorithm = 'SMOTENC',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'},
            data_dependents = {'categorical_features' : 'categoricals_indices'})
        self.smoteenn = Technique(
            name = 'smoteenn',
            module = 'imblearn.combine',
            algorithm = 'SMOTEENN',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        self.smotetomek = Technique(
            name = 'smotetomek',
            module = 'imblearn.combine',
            algorithm = 'SMOTETomek',
            defaults = {'sampling_strategy': 'auto'},
            runtimes = {'random_state': 'seed'})
        super().draft()
        return self