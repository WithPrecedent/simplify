"""
.. module:: siMpLify cookbook
  :synopsis: machine learning made simple
"""


from .cookbook import Cookbook
from .cookbook_step import CookbookStep
from .recipe import Recipe


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Cookbook',
           'CookbookStep',
           'Recipe']