"""
.. module:: siMpLify
  :synopsis: data science made simple
"""

import core
import cookbook
import almanac
from core.decorators import timer
from .core.menu import Menu
from .core.inventory import Inventory
from .core.ingredients import Ingredients

__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['core',
           'cookbook',
           'almanac',
           'timer',
           'Menu',
           'Inventory',
           'Ingredients']