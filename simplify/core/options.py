"""
.. module:: options
:synopsis: siMpLify base lexicon classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from numpy import datetime64
import pandas as pd
from pandas.api.types import CategoricalDtype

from simplify.core.utilities import listify



