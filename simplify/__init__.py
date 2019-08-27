"""
.. module:: siMpLify
  :synopsis: data science made simple
"""

from .decorators import timer
from .ingredients import Ingredients
from .inventory import Inventory
from .menu import Menu


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Ingredients',
           'Inventory',
           'Menu',
           'timer']