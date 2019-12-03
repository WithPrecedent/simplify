"""
.. module:: chef
:synopsis: algorithm and parameter builder
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from simplify.core.author import Author
from simplify.core.author import Content
from simplify.core.author import Outline
from simplify.core.utilities import listify


@dataclass
class Chef(Author):
    """Constructs Page instances from Outline instances for use in a Chapter.

    This class is a director for a complex content which constructs finalized
    algorithms with matching parameters. Because of the variance of supported
    packages and the nature of parameters involved (particularly data-dependent
    ones), the final construction of a Page is not usually completed until the
    'apply' method is called.

    Args:
        idea ('Idea'): an instance of Idea with user settings.
        content (Optional[Union['Author'], List['Author']]):
            instance(s) of Author subclass. Defaults to None.
        outline (Optional['Outline']): instance containing information
            needed to build the desired objects. Defaults to None.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'outline' and 'content' must also be passed. Defaults to True.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    idea: 'Idea'
    content: Optional[Union['Author'], List['Author']] = None
    outline: Optional['Outline'] = None
    auto_publish: Optional[bool] = True
    name: Optional[str] = None

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _build_conditional(self, 
            name: str, 
            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Modifies 'parameters' based upon various conditions.

        A subclass should have its own '_build_conditional' method for this
        method to modify 'parameters'. That method should have a 'parameters'
        argument and return the modified 'parameters'.

        Args:
            name (str): name of method being used.
            parameters (Dict[str, Any]): a dictionary of parameters.

        Returns:
            parameters (Dict[str, Any]): altered parameters based on condtions.

        """
        pass


@dataclass
class RecipeOutline(object):
    """Contains settings for creating a Algorithm and Parameters.
    
    Users can use the idiom 'x in Outline' to check if a particular attribute
    exists and is not None. This means default values for optional arguments
    should generally be set to None to allow use of that idiom.
    
    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. 
        module (Optional[str]): name of module where object to incorporate is
            located (can either be a siMpLify or non-siMpLify object). Defaults
            to None.
        algorithm (Optional[str]): name of object within 'module' to load.
            Defaults to None.
            
    """
    name: Optional[str] = 'outline'
    module: Optional[str] = None
    algorithm: Optional[str] = None
    default: Optional[Dict[str, Any]] = None
    required: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, str]] = None
    data_dependent: Optional[Dict[str, str]] = None
    selected: Optional[Union[bool, List[str]]] = False
    conditional: Optional[bool] = False
    hyperparameter_search: Optional[bool] = False
    critic_dependent: Optional[Dict[str, str]] = None
    export_file: Optional[str] = None
 