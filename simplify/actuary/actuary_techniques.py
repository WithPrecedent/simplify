"""
.. module:: chef techniques
:synopsis: default techniques for chef subpackage
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections import namedtuple

fields = [
    'name', 'module', 'algorithm', 'default_parameters', 'extra_parameters',
    'runtime_parameters', 'selected_parameters', 'conditional_parameters',
    'data_dependent']
Technique = namedtuple('technique', fields, default = (None,) * len(fields))

""" Summary Techniques """

summary_count = Technique(
    name = 'count',
    module = 'numpy.ndarray',
    algorithm = 'size')
summary_min = Technique(
    name = 'minimum',
    module = 'numpy',
    algorithm = 'nanmin')
summary_q1 = Technique(
    name = 'quantile1',
    module = 'numpy',
    algorithm = 'nanquantile',
    extra_parameters = {'q': 0.25})
summary_median = Technique(
    name = 'median',
    module = 'numpy',
    algorithm = 'nanmedian')
summary_q3 = Technique(
    name = 'quantile3',
    module = 'numpy',
    algorithm = 'nanquantile',
    extra_parameters = {'q': 0.25})
summary_max = Technique(
    name = '',
    module = 'numpy',
    algorithm = 'nanmax')
summary_mad = Technique(
    name = 'median absoluate deviation',
    module = 'scipy.stats',
    algorithm = 'median_absolute_deviation',
    extra_parameters = {'nan_policy': 'omit'})
summary_mean = Technique(
    name = 'mean',
    module = 'numpy',
    algorithm = 'nanmean')
summary_std = Technique(
    name = 'standard deviation',
    module = 'numpy',
    algorithm = 'nanstd')
summary_standard_error = Technique(
    name = 'standard_error',
    module = 'scipy.stats',
    algorithm = 'sem',
    extra_parameters = {'nan_policy': 'omit'})
summary_geometric_mean = Technique(
    name = 'geometric_mean',
    module = 'scipy.stats',
    algorithm = 'gmean')
summary_geometric_std = Technique(
    name = 'geometric_standard_deviation',
    module = 'scipy.stats',
    algorithm = 'gstd')
summary_harmonic_mean = Technique(
    name = 'harmonic_mean',
    module = 'scipy.stats',
    algorithm = 'hmean')
summary_mode = Technique(
    name = 'mode',
    module = 'scipy.stats',
    algorithm = 'mode',
    extra_parameters = {'nan_policy': 'omit'})
summary_sum = Technique(
    name = 'sum',
    module = 'numpy',
    algorithm = 'nansum')
summary_kurtosis = Technique(
    name = 'kurtosis',
    module = 'scipy.stats',
    algorithm = 'kurtosis',
    extra_parameters = {'nan_policy': 'omit'})
summary_skew = Technique(
    name = 'skew',
    module = 'scipy.stats',
    algorithm = 'skew',
    extra_parameters = {'nan_policy': 'omit'})
summary_variance = Technique(
    name = 'variance',
    module = 'numpy',
    algorithm = 'nanvar')
summary_variation = Technique(
    name = 'variation',
    module = 'scipy.stats',
    algorithm = 'variation',
    extra_parameters = {'nan_policy': 'omit'})
summary_unique = Technique(
    name = 'unique_values',
    module = 'numpy',
    algorithm = 'nunique')
