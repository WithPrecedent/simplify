"""
.. module:: simplify
  :synopsis: data science made simple
"""

from .ingredients import Ingredients
from .implements.decorators import timer
from .implements.inventory import Inventory
from .implements.menu import Menu


__version__ = '0.1.0'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Ingredients',
           'Inventory',
           'Menu',
           'timer']