"""
explorer.components:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify
from . import base


@dataclasses.dataclass
class Test(SimpleDirector):
    """Applies statistical tests to data.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
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
            'y_true': getattr(recipe.dataset, 'y_' + self.data_to_review),
            'y_pred': recipe.predictions}
        super().implement()
        return self
    
    
@dataclasses.dataclass
class Summarize(SimpleDirector):
    """Summarizes data.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    name: str = 'summarizer'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets options for Summarize class."""
        super().draft()
        self._options = SimpleRepository(contents = {
            'count': Option(
                name = 'count',
                module = 'numpy.ndarray',
                algorithm = 'size'),
            'min': Option(
                name = 'minimum',
                module = 'numpy',
                algorithm = 'nanmin'),
            'q1': Option(
                name = 'quantile1',
                module = 'numpy',
                algorithm = 'nanquantile',
                required = {'q': 0.25}),
            'median': Option(
                name = 'median',
                module = 'numpy',
                algorithm = 'nanmedian'),
            'q3': Option(
                name = 'quantile3',
                module = 'numpy',
                algorithm = 'nanquantile',
                required = {'q': 0.25}),
            'max': Option(
                name = '',
                module = 'numpy',
                algorithm = 'nanmax'),
            'mad': Option(
                name = 'median absoluate deviation',
                module = 'scipy.stats',
                algorithm = 'median_absolute_deviation',
                required = {'nan_policy': 'omit'}),
            'mean': Option(
                name = 'mean',
                module = 'numpy',
                algorithm = 'nanmean'),
            'std': Option(
                name = 'standard deviation',
                module = 'numpy',
                algorithm = 'nanstd'),
            'standard_error': Option(
                name = 'standard_error',
                module = 'scipy.stats',
                algorithm = 'sem',
                required = {'nan_policy': 'omit'}),
            'geometric_mean': Option(
                name = 'geometric_mean',
                module = 'scipy.stats',
                algorithm = 'gmean'),
            'geometric_std': Option(
                name = 'geometric_standard_deviation',
                module = 'scipy.stats',
                algorithm = 'gstd'),
            'harmonic_mean': Option(
                name = 'harmonic_mean',
                module = 'scipy.stats',
                algorithm = 'hmean'),
            'mode': Option(
                name = 'mode',
                module = 'scipy.stats',
                algorithm = 'mode',
                required = {'nan_policy': 'omit'}),
            'sum': Option(
                name = 'sum',
                module = 'numpy',
                algorithm = 'nansum'),
            'kurtosis': Option(
                name = 'kurtosis',
                module = 'scipy.stats',
                algorithm = 'kurtosis',
                required = {'nan_policy': 'omit'}),
            'skew': Option(
                name = 'skew',
                module = 'scipy.stats',
                algorithm = 'skew',
                required = {'nan_policy': 'omit'}),
            'variance': Option(
                name = 'variance',
                module = 'numpy',
                algorithm = 'nanvar'),
            'variation': Option(
                name = 'variation',
                module = 'scipy.stats',
                algorithm = 'variation',
                required = {'nan_policy': 'omit'}),
            'unique': Option(
                name = 'unique_values',
                module = 'numpy',
                algorithm = 'nunique')}
        return self

