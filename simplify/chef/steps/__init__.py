"""
.. module:: siMpLify cookbook steps
  :synopsis: machine learning workflow made simple
"""

from .cleave import Cleave
from .encode import Encode
from .mix import Mix
from .model import Model
from .reduce import Reduce
from .sample import Sample
from .scale import Scale
from .split import Split


__all__ = ['Cleave',
           'Encode',
           'Mix',
           'Model',
           'Reduce',
           'Sample',
           'Scale',
           'Split']
