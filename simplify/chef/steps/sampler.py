"""
.. module:: sampler
:synopsis: synthetically resamples data to different distributions
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.creator.typesetter import SimpleDirector
from simplify.creator.typesetter import Outline


@dataclass
class Sampler(SimpleDirector):
    """Resamples data based on outcome distributions.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    name: str = 'sampler'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self._options = SimpleOptions(options = {
            'adasyn': Outline(
                name = 'adasyn',
                module = 'imblearn.over_sampling',
                algorithm = 'ADASYN',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'cluster': Outline(
                name = 'cluster',
                module = 'imblearn.under_sampling',
                algorithm = 'ClusterCentroids',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'knn': Outline(
                name = 'knn',
                module = 'imblearn.under_sampling',
                algorithm = 'AllKNN',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'near_miss': Outline(
                name = 'near_miss',
                module = 'imblearn.under_sampling',
                algorithm = 'NearMiss',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'random_over': Outline(
                name = 'random_over',
                module = 'imblearn.over_sampling',
                algorithm = 'RandomOverSampler',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'random_under': Outline(
                name = 'random_under',
                module = 'imblearn.under_sampling',
                algorithm = 'RandomUnderSampler',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'smote': Outline(
                name = 'smote',
                module = 'imblearn.over_sampling',
                algorithm = 'SMOTE',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'smotenc': Outline(
                name = 'smotenc',
                module = 'imblearn.over_sampling',
                algorithm = 'SMOTENC',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'},
                data_dependent = {
                    'categorical_features': 'categoricals_indices'}),
            'smoteenn': Outline(
                name = 'smoteenn',
                module = 'imblearn.combine',
                algorithm = 'SMOTEENN',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'}),
            'smotetomek': Outline(
                name = 'smotetomek',
                module = 'imblearn.combine',
                algorithm = 'SMOTETomek',
                default = {'sampling_strategy': 'auto'},
                runtime = {'random_state': 'seed'})}
        return self