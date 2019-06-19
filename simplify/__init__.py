"""
.. module:: simplify
  :synopsis: data science made simple
"""

from .ingredients import Ingredients
from .menu import Menu
from .inventory import Inventory
from .cookbook.timer import timer


__version__ = '0.1.0'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Ingredients',
           'Menu',
           'Inventory',
           'timer']