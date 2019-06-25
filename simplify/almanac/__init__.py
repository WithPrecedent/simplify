"""
.. module:: simplify_almanac
  :synopsis: data preparation made simple
"""


from .almanac import Almanac
from .stages import Bale
from .stages import Clean
from .stages import Cultivate
from .stages import Reap
from .stages import Thresh



__version__ = '0.1.0'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Almanac',
           'Bale',
           'Clean',
           'Cultivate',
           'Reap',
           'Thresh']