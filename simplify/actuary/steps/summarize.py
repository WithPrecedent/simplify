"""
.. module:: summarize
:synopsis: summarizes data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simplify.core.technique import SimpleComposer
from simplify.core.technique import SimpleDesign


@dataclass
class Summarize(SimpleComposer):
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
        options = {
            'count': SimpleDesign(
                name = 'count',
                module = 'numpy.ndarray',
                algorithm = 'size'),
            'min': SimpleDesign(
                name = 'minimum',
                module = 'numpy',
                algorithm = 'nanmin'),
            'q1': SimpleDesign(
                name = 'quantile1',
                module = 'numpy',
                algorithm = 'nanquantile',
                required = {'q': 0.25}),
            'median': SimpleDesign(
                name = 'median',
                module = 'numpy',
                algorithm = 'nanmedian'),
            'q3': SimpleDesign(
                name = 'quantile3',
                module = 'numpy',
                algorithm = 'nanquantile',
                required = {'q': 0.25}),
            'max': SimpleDesign(
                name = '',
                module = 'numpy',
                algorithm = 'nanmax'),
            'mad': SimpleDesign(
                name = 'median absoluate deviation',
                module = 'scipy.stats',
                algorithm = 'median_absolute_deviation',
                required = {'nan_policy': 'omit'}),
            'mean': SimpleDesign(
                name = 'mean',
                module = 'numpy',
                algorithm = 'nanmean'),
            'std': SimpleDesign(
                name = 'standard deviation',
                module = 'numpy',
                algorithm = 'nanstd'),
            'standard_error': SimpleDesign(
                name = 'standard_error',
                module = 'scipy.stats',
                algorithm = 'sem',
                required = {'nan_policy': 'omit'}),
            'geometric_mean': SimpleDesign(
                name = 'geometric_mean',
                module = 'scipy.stats',
                algorithm = 'gmean'),
            'geometric_std': SimpleDesign(
                name = 'geometric_standard_deviation',
                module = 'scipy.stats',
                algorithm = 'gstd'),
            'harmonic_mean': SimpleDesign(
                name = 'harmonic_mean',
                module = 'scipy.stats',
                algorithm = 'hmean'),
            'mode': SimpleDesign(
                name = 'mode',
                module = 'scipy.stats',
                algorithm = 'mode',
                required = {'nan_policy': 'omit'}),
            'sum': SimpleDesign(
                name = 'sum',
                module = 'numpy',
                algorithm = 'nansum'),
            'kurtosis': SimpleDesign(
                name = 'kurtosis',
                module = 'scipy.stats',
                algorithm = 'kurtosis',
                required = {'nan_policy': 'omit'}),
            'skew': SimpleDesign(
                name = 'skew',
                module = 'scipy.stats',
                algorithm = 'skew',
                required = {'nan_policy': 'omit'}),
            'variance': SimpleDesign(
                name = 'variance',
                module = 'numpy',
                algorithm = 'nanvar'),
            'variation': SimpleDesign(
                name = 'variation',
                module = 'scipy.stats',
                algorithm = 'variation',
                required = {'nan_policy': 'omit'}),
            'unique': SimpleDesign(
                name = 'unique_values',
                module = 'numpy',
                algorithm = 'nunique')}
        self.options = SimpleOptions(options = options, parent = self)
        return self

