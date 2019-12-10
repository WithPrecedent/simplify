"""
.. module:: inventory
:synopsis: resource management made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
import datetime
import glob
import os
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

from simplify.library.defaults import Defaults
from simplify.creator.typesetter import CodexOptions
from simplify.library.utilities import listify


@dataclass
class Inventory(CodexOptions):

    def __post_init__(self):
        return self

    def draft(self):
        self.data_folders = {}
        self.results_folders = {}




@dataclass
class GraphicsFiler(SimpleFile):

    folder_path: str
    file_name: str
    file_format: 'FileFormat'

    def __post_init__(self):
        return self


