"""
.. module:: tests
:synopsis: exploratory data tests
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.contributor import SimpleContributor
from simplify.core.contributor import Outline


@dataclass
class Test(SimpleContributor):
    """Applies statistical tests to data.

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
    name: Optional[str] = 'tester'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        options = {
            'ks_distribution': ['scipy.stats', 'ks_2samp'],
            'ks_goodness': ['scipy.stats', 'kstest'],
            'kurtosis_test': ['scipy.stats', 'kurtosistest'],
            'normal': ['scipy.stats', 'normaltest'],
            'pearson': ['scipy.stats.pearsonr']}
        return self

    def publish(self):
        self.runtime_parameters = {
            'y_true': getattr(recipe.ingredients, 'y_' + self.data_to_review),
            'y_pred': recipe.predictions}
        super().implement()
        return self