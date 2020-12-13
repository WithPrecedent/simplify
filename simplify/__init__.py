"""
.. module:: siMpLify
:synopsis: data science made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from simplify.core.dataset import Dataset
from simplify.core.clerk import Clerk
from simplify.core.idea import Idea
from simplify.core.project import Project
from simplify.core.worker import Worker
from simplify.core.utilities import simple_timer


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = [
    'Dataset',
    'Clerk',
    'Idea',
    'Project',
    'Worker',
    'simple_timer']