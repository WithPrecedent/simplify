"""
.. module:: siMpLify analyst
:synopsis: siMpLify analyst package
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

from simplify.analyst.analyst import Analyst
from simplify.analyst.analyst import Cookbook
from simplify.analyst.analyst import Tools


__all__ = [
    'Analyst',
    'Cookbook',
    'Tools']

COMPONENTS = {
    'worker': 'Analyst',
    'book': 'Cookbook',
    'catalog': 'Tools'}