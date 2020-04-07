"""
.. module:: siMpLify
:synopsis: data science made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from simplify.core.base import SimpleSystem
from simplify.core.base import SimpleCreator
from simplify.core.base import SimpleSimpleRepository
from simplify.core.base import SimpleRepository
from simplify.core.base import SimpleComponent
from simplify.core.dataset import Dataset
from simplify.core.filer import Filer
from simplify.core.idea import Idea
from simplify.core.project import Project
from simplify.core.utilities import simple_timer


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = [
    'SimpleSystem',
    'SimpleCreator',
    'SimpleSimpleRepository',
    'SimpleRepository',
    'SimpleComponent',
    'Dataset',
    'Filer',
    'Idea',
    'Project',
    'timer']