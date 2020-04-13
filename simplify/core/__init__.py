"""
.. module:: core
:synopsis: siMpLify base classes and utilities
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from simplify.core.component import SimpleComponent
from simplify.core.component import SimpleLoader
from simplify.core.component import SimplePlan
from simplify.core.component import SimpleProxy
from simplify.core.component import SimpleRepository
from simplify.core.handler import SimpleHandler
from simplify.core.handler import SimpleParallel
from simplify.core.system import SimpleSystem


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = [
    'SimpleComponent',
    'SimplePlan',
    'SimpleProxy',
    'SimpleRepository'
    'SimpleHandler',
    'SimpleParallel',
    'SimpleSystem']
