"""
.. module:: summarize
:synopsis: summarizes data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

from simplify.core.typesetter import SimpleDirector
from simplify.core.typesetter import Outline


@dataclass
class Summarize(SimpleDirector):
    """Summarizes data.

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
    name: str = 'summarizer'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets options for Summarize class."""
        super().draft()
        self._options = SimpleOptions(options = {
            'count': Outline(
                name = 'count',
                module = 'numpy.ndarray',
                algorithm = 'size'),
            'min': Outline(
                name = 'minimum',
                module = 'numpy',
                algorithm = 'nanmin'),
            'q1': Outline(
                name = 'quantile1',
                module = 'numpy',
                algorithm = 'nanquantile',
                required = {'q': 0.25}),
            'median': Outline(
                name = 'median',
                module = 'numpy',
                algorithm = 'nanmedian'),
            'q3': Outline(
                name = 'quantile3',
                module = 'numpy',
                algorithm = 'nanquantile',
                required = {'q': 0.25}),
            'max': Outline(
                name = '',
                module = 'numpy',
                algorithm = 'nanmax'),
            'mad': Outline(
                name = 'median absoluate deviation',
                module = 'scipy.stats',
                algorithm = 'median_absolute_deviation',
                required = {'nan_policy': 'omit'}),
            'mean': Outline(
                name = 'mean',
                module = 'numpy',
                algorithm = 'nanmean'),
            'std': Outline(
                name = 'standard deviation',
                module = 'numpy',
                algorithm = 'nanstd'),
            'standard_error': Outline(
                name = 'standard_error',
                module = 'scipy.stats',
                algorithm = 'sem',
                required = {'nan_policy': 'omit'}),
            'geometric_mean': Outline(
                name = 'geometric_mean',
                module = 'scipy.stats',
                algorithm = 'gmean'),
            'geometric_std': Outline(
                name = 'geometric_standard_deviation',
                module = 'scipy.stats',
                algorithm = 'gstd'),
            'harmonic_mean': Outline(
                name = 'harmonic_mean',
                module = 'scipy.stats',
                algorithm = 'hmean'),
            'mode': Outline(
                name = 'mode',
                module = 'scipy.stats',
                algorithm = 'mode',
                required = {'nan_policy': 'omit'}),
            'sum': Outline(
                name = 'sum',
                module = 'numpy',
                algorithm = 'nansum'),
            'kurtosis': Outline(
                name = 'kurtosis',
                module = 'scipy.stats',
                algorithm = 'kurtosis',
                required = {'nan_policy': 'omit'}),
            'skew': Outline(
                name = 'skew',
                module = 'scipy.stats',
                algorithm = 'skew',
                required = {'nan_policy': 'omit'}),
            'variance': Outline(
                name = 'variance',
                module = 'numpy',
                algorithm = 'nanvar'),
            'variation': Outline(
                name = 'variation',
                module = 'scipy.stats',
                algorithm = 'variation',
                required = {'nan_policy': 'omit'}),
            'unique': Outline(
                name = 'unique_values',
                module = 'numpy',
                algorithm = 'nunique')}
        return self

