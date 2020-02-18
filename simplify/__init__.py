"""
.. module:: siMpLify
:synopsis: data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from simplify.core import startup
from simplify.core.project import ProjectManager
from simplify.core.utilities import timer


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = [
    'startup',
    'ProjectManager',
    'timer']