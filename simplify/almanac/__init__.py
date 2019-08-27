"""
.. module:: siMpLify almanac
  :synopsis: data preparation made simple
"""

from .almanac import Almanac
from .almanac_step import AlmanacStep
from .plan import Plan


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Almanac',
           'AlmanacStep',
           'Plan']