"""
.. module:: simplify_stages
  :synopsis: data processing workflow made simple
"""

from .bale import Bale
from .clean import Clean
from .cultivate import Cultivate
from .reap import Reap
from .thresh import Thresh



__version__ = '0.1.0'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Bale',
           'Clean',
           'Cultivate',
           'Reap',
           'Thresh']
