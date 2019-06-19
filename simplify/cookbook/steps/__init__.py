"""
.. module:: simplify_steps
  :synopsis: machine learning workflow made simple
"""

from .cleave import Cleave
from .custom import Custom
from .encode import Encode
from .mix import Mix
from .model import Model
from .reduce import Reduce
from .sample import Sample
from .scale import Scale
from .split import Split


__version__ = '0.1.0'

__author__ = 'Corey Rayburn Yung'

__all__ = ['Cleave',
           'Custom',
           'Encode',
           'Mix',
           'Model',
           'Reduce',
           'Sample',
           'Scale',
           'Split']
