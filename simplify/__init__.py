"""
.. module:: siMpLify
  :synopsis: data science made simple
"""

import core
from core.decorators import timer
from core.ingredients import Ingredients
from core.inventory import Inventory
from core.menu import Menu


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['core',
           'Ingredients',
           'Inventory',
           'Menu',
           'timer']