"""
.. module:: simplify_almanac
  :synopsis: data preparation made simple
"""

from .almanac import Almanac
from .instructions import Instructions
from .stages import Sow
from .stages import Harvest
from .stages import Clean
from .stages import Bundle
from .stages import Deliver


__version__ = '0.1.0'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Almanac',
           'Instructions',
           'Sow',
           'Harvest',
           'Clean',
           'Bundle',
           'Deliver']