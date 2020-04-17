"""
.. module:: core
:synopsis: siMpLify base classes
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from simplify.core.component import SimpleComponent
from simplify.core.component import SimpleProxy
from simplify.core.configuration import SimpleSettings
from simplify.core.definition import SimpleType
from simplify.core.handler import SimpleHandler
from simplify.core.loader import SimpleLoader
from simplify.core.plan import SimplePlan
from simplify.core.repository import SimpleRepository
from simplify.core.system import SimpleSystem
from simplify.core.technique import SimpleTechnique


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = [
    'SimpleComponent',
    'SimpleProxy',
    'SimpleSettings',
    'SimpleType',
    'SimpleHandler',
    'SimpleLoader',
    'SimplePlan',
    'SimpleRepository',
    'SimpleSystem',
    'SimpleTechnique']
