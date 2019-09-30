"""
.. module:: siMpLify
:synopsis: data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from simplify.idea import Idea
from simplify.depot import Depot
from simplify.ingredients import Ingredients
from simplify.__main__ import Simplify
from simplify.core.decorators import timer


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Idea',
           'Depot',
           'Ingredients',
           'Simplify',
           'timer']