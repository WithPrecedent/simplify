"""
.. module:: step
:synopsis: a step in a siMpLify iterable.
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, List, Dict, Union, Tuple

import numpy as np
import pandas as pd

from simplify.core.base import SimpleClass
from simplify.core.composer import SimpleComposer
from simplify.core.ingredients import Ingredients
from simplify.core.technique import SimpleTechnique
from simplify.core.utilities import listify


@dataclass
class SimpleStep(SimpleClass):
    """Base class for building and controlling iterable techniques and/or
    other packages.

    This class adds methods useful to create iterators and iterate over passed
    arguments based upon user-selected options. SimplePackage subclasses
    construct iterators and process data with those iterators.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        techniques (list or str): names of techniques to be applied. These names
            should match keys in the 'options' attribute.

    It is also a child class of SimpleClass. So, its documentation applies as
    well.

    """
    name: str = 'simple_step'
    techniques: object = None
    
    def __post_init__(self) -> None:
        super().__post_init__()
        return self
    
    """ Private Methods """
    
    def _get_conditional(self, parameters: Dict) -> Dict:
        """Modifies 'parameters' based upon various conditions.

        A subclass should have its own '_get_conditional' method for this
        method to modify 'parameters'. That method should have a 'parameters'
        argument and return the modified 'parameters'.
        
        Args:
            parameters (Dict): a dictionary of parameters.
            
        Returns:
            parameters (Dict): altered parameters based on condtions.

        """
        pass
        
    """ Core siMpLify Methods """
        
    def draft(self):
        """ Subclass should provide their own 'options' stored in this format:
    
            self.options = {string_key, SimpleDesign}
            
        """
        super().draft()
        return self
    
    def publish(self, data: Union[Ingredients, Tuple]):
        self.composer = SimpleComposer(
            options = self.options,
            techniques = self.techniques)
        for technique in listify(self._convert_wildcards(self.techniques)):
            self.options[technique] = self.composer.publish(
                technique = technique,
                data = data,
                step = self)
        return self


@dataclass
class SimpleDesign(object):
    """Contains settings for creating a SimpleAlgorithm. """

    name: str = 'simple_design'
    step: str = ''
    module: str = None
    algorithm: str = None
    default: object = None
    required: object = None
    runtime: object = None
    data_dependent: object = None
    selected: Union[bool, List] = False
    conditional: bool = False
    hyperparameter_search: bool = False
