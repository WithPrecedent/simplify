"""
.. module:: siMpLify core
 :synopsis: siMpLify base classes and functions
"""

from simplify.core.base import (SimpleClass, SimpleManager, SimplePlan,
                                SimpleStep)
from simplify.core.decorators import timer

__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['SimpleClass',
           'SimpleManager',
           'SimplePlan',
           'SimpleStep',
           'timer']