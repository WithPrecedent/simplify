"""
.. module:: siMpLify
  :synopsis: data science made simple
"""

from .core.base import Idea, Depot, Ingredients
from .core.decorators import timer

__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Idea',
           'Depot',
           'Ingredients'
           'timer',]