"""
.. module:: technique
:synopsis: siMpLify algorithms and parameters
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
from inspect import signature
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y

from simplify.core.repository import Outline


@dataclass
class TechniqueOutline(Outline):
    """Contains settings for creating a Technique instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify object).
        algorithm: str = None
        default: Optional[Dict[str, Any]] = field(default_factory = dict)
        required: Optional[Dict[str, Any]] = field(default_factory = dict)
        runtime: Optional[Dict[str, str]] = field(default_factory = dict)
        selected: Optional[Union[bool, List[str]]] = False
        conditional: Optional[bool] = False
        data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)

    """
    name: str
    module: str
    algorithm: str = None
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)
    fit_method: Optional[str] = field(default_factory = lambda: 'fit')
    transform_method: Optional[str] = field(
        default_factory = lambda: 'transform')


