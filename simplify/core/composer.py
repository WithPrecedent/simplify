"""
.. module:: compose
:synopsis: creates siMpLify-compatible algorithms
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module

Algorithm = SimpleAlgorithm
Technique = SimpleTechnique

@dataclass
class SimpleComposer(SimpleClass):
    """Creates siMpLify-compatible algorithms.

    Args:
        name (str): public name of class, used by various methods and classes
            throughout the siMpLify package.
    """
    name: str = 'generic_composer'

    def __post_init__(self):
        super().__post_init__()
        # Declares parameter_types.
        self.parameter_types = (
            'idea',
            'selected',
            'extra',
            'runtime',
            'conditional')
        # Create dictionary of imported modules with keys as technique names and
        # values as the imported module.
        self.options = {}
        return self

    """ Private Methods """

    def _add_gpu_techniques(self):
        """
        Subclasses should provide their own '_add_gpu_techniques' method if
        there are appropriate techniques which use a local gpu.
        """
        pass

    def _get_algorithm(self, technique: SimpleTechnique):
        """Acquires algorithm from 'options' dict or imports appropriate module.

        This method looks in the 'options' dict first to avoiding unncessary
        reimporting of modules. If 'technique.name' does not match a key in
        'options', a new object is imported from a module and that is then
        added to the 'options' dict.

        Args:
            technique (Technique): object containing configuration information
                for an Algorithm to be created.

        Returns:
            Algorithm object configured appropriately.

        """
        try:
            return self.options[technique.name]
        except KeyError:
            algorithm = getattr(import_module(technique.module),
                                technique.algorithm)
            self.options[technique.name] = algorithm
            return algorithm

    def _get_parameters(self, technique: Technique):
        """Calls appropriate methods for constructing technique parameters.

        Args:
            technique (Technique): object containing configuration information
                for parameters to be constructed.

        Returns:
            dict containing parameters for the technique. Data dependent
                parameters are not incorporated at this stage.

        """
        for parameter_type in self.parameter_types:
            parameters = getattr(
                self, ''.join(['_get_', parameter_type, 's']))(
                    technique = technique,
                    parameters = parameters)
        return parameters

    def _get_ideas(self, technique: SimpleTechnique, parameters: dict):
        """[summary]

        Args:
            technique (Technique): object containing configuration information
                for parameters to be constructed.
            parameters (dict): [description]
        """
        if parameters:
            return parameters
        else:
            try:
                parameters = self.idea[''.join([technique.name, '_parameters'])]
            except KeyError:
                try:
                    parameters = self.idea[''.join([self.name, '_parameters'])]
                except KeyError:
                    pass
        return parameters

    def _get_selecteds(self, technique: SimpleTechnique, parameters: dict):
        """[summary]

        Args:
            technique (Technique): object containing configuration information
                for parameters to be constructed.
            parameters (dict): [description]

        Returns:
            [type]: [description]
        """
        if technique.selected:
            parameters_to_use = list(
                technique.defaults.keys())
            new_parameters = {}
            for key, value in parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            parameters = new_parameters
        return parameters

    def _get_extras(self, technique: SimpleTechnique, parameters: dict):
        """Adds extra parameters (mandatory additions) to parameters.

        Args:
            technique (Technique): object containing configuration information
                for parameters to be constructed.
            parameters (dict): [description]
        """
        try:
            parameters.update(technique.extras)
        except TypeError:
            pass
        return parameters

    def _get_runtimes(self, technique: SimpleTechnique, parameters: dict):
        """[summary]

        Args:
            technique (Technique): object containing configuration information
                for parameters to be constructed.
            parameters (dict): [description]
        """
        try:
            for key, value in technique.runtimes.items():
                try:
                    parameters.update({key: getattr(self, value)})
                except AttributeError:
                    error = ' '.join('no matching runtime parameter',
                                     key, 'found in', self.name)
                    raise AttributeError(error)
        except TypeError:
            pass
        return parameters

    def _get_conditionals(self, technique: SimpleTechnique, parameters: dict):
        """[summary]

        Args:
            technique (Technique): object containing configuration information
                for parameters to be constructed.
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
            technique (Technique): object containing configuration information
                for parameters to be constructed.
            parameters (dict, optional): [description]. Defaults to None.
        """
        if technique in ['none', 'None', None]:
            return None
        else:
            technique = getattr(self, '_'.join([step, technique]))
            algorithm = self._get_algorithm(technique = technique)
            parameters = _get_parameters(
                technique = technique,
                parameters = parameters)
            return Algorithm(
                technique = technique.name,
                algorithm = algorithm,
                parameters = parameters,
                data_dependents = technique.data_dependents)


@dataclass
class SimpleTechnique(object):

    name: str = 'generic_technique'
    module: str = None
    algorithm: str = None
    defaults: object = None
    extras: object = None
    runtimes: object = None
    data_dependents: object = None
    selected: bool = False
    conditional: bool = False


@dataclass
class SimpleAlgorithm(SimpleClass):
    """[summary]

    Args:
        object ([type]): [description]
    """

    technique: str
    algorithm: object
    parameters: object
    data_dependents: object = None

    def __post_init__(self):
        self.draft()
        return self

    """ Private Methods """

    def _add_parameters(self):
        """[summary]
        """
        try:
            self.algorithm = self.algorithm(**self.parameters)
        except AttributeError:
            try:
                self.algorithm = self.algorithm(self.parameters)
        return self

    def _add_data_dependents(self, ingredients: Ingredients):
        """[summary]

        Args:
            ingredients (Ingredients): [description]
        """
        for key, value in self.datas.items():
            self.parameters.update({key, getattr(ingredients, value)})
        self._add_parameters()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """[summary]
        """
        self.options = (
            [i for i in list(self.__dict__.keys()) if isinstance(i, Technique)])
        if not self.data_dependents:
            self._add_parameterss()
        return self

    def publish(self, variable, limitations, *args, **kwargs):
        """Subclasses should provide their own 'publish' methods."""
        if self.data_dependents:
            self._add_data_dependents()
        return self

    """ Properties """
    
    @property
    def options(self):
        return {k: v for (k, v) in self.__dict__.items() if isinstance(v,
                    Technique)}
