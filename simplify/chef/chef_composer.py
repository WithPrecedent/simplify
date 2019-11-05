"""
.. module:: chef composer
:synopsis: creates siMpLify-compatible algorithms for chef subpackage
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module

from simplify.chef.search_composer import SearchComposer
from simplify.core.compose import SimpleAlgorithm
from simplify.core.compose import SimpleComposer
from simplify.core.compose import SimpleTechnique
from simplify.core.decorators import numpy_shield


@dataclass
class ChefComposer(SimpleComposer):
    """[summary]

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'chef_composer'

    def __post_init__(self):
        super().__post_init__()
        # Declares parameter_types.
        self.parameter_types = (
            'idea',
            'selected',
            'extra',
            'search'
            'runtime',
            'conditional')
        return self

    """ Private Methods """

    def _get_search(self, technique: SimpleTechnique, parameters: dict):
        """[summary]

        Args:
            technique (SimpleTechnique): [description]
            parameters (dict): [description]

        """
        return parameters

    """ Core siMpLify Methods """

    def draft(self):
        """[summary]
        """
        # Subclasses should create Technique instances here.
        if self.gpu:
            self.add_gpu_techniques()
        return self

    def publish(self, technique: str, parameters: dict = None):
        """[summary]

        Args:
            technique (str): [description]
            parameters (dict, optional): [description]. Defaults to None.
        """
        if technique in ['none', 'None', None]:
            return None
        else:
            technique = getattr(self, '_'.join([step, technique]))
            algorithm = self._get_algorithm(technique = technique)
            parameters = self._get_parameters(
                technique = technique,
                parameters = parameters)
            return ChefAlgorithm(
                technique = technique.name,
                algorithm = algorithm,
                parameters = parameters,
                data_dependents = technique.data_dependents,
                hyperparameter_search = technique.hyperparameter_search,
                space = self.space)


@dataclass
class ChefTechnique(SimpleTechnique):

    name: str = 'chef_technique'
    module: str = None
    algorithm: str = None
    defaults: object = None
    extras: object = None
    runtimes: object = None
    data_dependents: object = None
    selected: bool = False
    conditional: bool = False
    hyperparameter_search: bool = False


