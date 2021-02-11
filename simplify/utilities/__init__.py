"""
utilities: sourdough utility functions and decorators
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    decorators: class, method, and function decorators.
    memory: classes and functions for conserving system memory.
    tools: assorted functions for making basic tasks easier.

ToDo:
    Add lazy import system to the 'utilities' subpackage.
    
"""

__version__ = '0.1.3'

__author__ = 'Corey Rayburn Yung'


from .decorators import *
from .memory import *
from .tools import *
