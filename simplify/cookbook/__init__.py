"""
.. module:: simplify_cookbook
  :synopsis: machine learning made simple
"""


from .cookbook import Cookbook
from .recipe import Recipe
from .timer import timer
from ..ingredients import Ingredients



__version__ = '0.1.0'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Cookbook',
           'Ingredients',
           'Recipe',
           'timer']