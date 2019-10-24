"""
.. module:: tests
:synopsis: tests of core siMpLify components
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import os
import sys
sys.path.insert(0, os.path.join('..', 'simplify'))
sys.path.insert(0, os.path.join('..', '..', 'simplify'))

import simplify.builder as builder

algorithm, parameters = builder.create(
    configuration = {'general': {'gpu': True, 'seed': 4}},
    package = 'chef',
    step = 'scale',
    technique = 'normalize',
    parameters = {'copy': False})

print(algorithm, parameters)


