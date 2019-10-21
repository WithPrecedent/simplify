
"""
.. module:: techniques
:synopsis: default techniques for siMpLify package
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections import namedtuple

Technique = namedtuple('name',
    ['module', 'algorithm', 'parameters', 'default_parameters',
     'extra_parameters', 'runtime_parameters'])

""" Scale Techniques """

scale_bins = Technique(
    name = 'bins', module = 'sklearn.preprocessing',
    algorithm = 'KBinsDiscretizer')
scale_gauss = Technique(
    name = 'gauss', module = 'simplify.chef.steps.techniques.gaussify',
    algorithm = 'Gaussify')
scale_maxabs = Technique(
    name = 'maxabs', module = 'sklearn.preprocessing',
    algorithm = 'MaxAbsScaler')
scale_minmax = Technique(
    name = 'minmax', module = 'sklearn.preprocessing',
    algorithm = 'MinMaxScaler')
scale_normalize = Technique(
    name = 'normalize', module = 'sklearn.preprocessing',
    algorithm = 'Normalizer')
scale_quantile = Technique(
    name = 'quantile', module = 'sklearn.preprocessing',
    algorithm = 'QuantileTransformer')
scale_robust = Technique(
    name = 'robust', module = 'sklearn.preprocessing',
    algorithm = 'RobustScaler')
scale_standard = Technique(
    name = 'standard', module = 'sklearn.preprocessing',
    algorithm = 'StandardScaler')