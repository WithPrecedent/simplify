"""
.. module:: composer
:synopsis: turns techniques into siMpLify-compatible algorithms
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module


@dataclass
class SimpleComposer(SimpleClass):
    """Creates siMpLify-compatible algorithms.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
            
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
        # Initializees dictionary of imported modules with keys as technique
        # names and values as imported objects.
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
            technique (SimpleTechnique): object containing configuration 
                information for an Algorithm to be created.

        Returns:
            Algorithm object configured appropriately.

        """
        try:
            return self.options[technique.name]
        except KeyError:
            algorithm = getattr(
                import_module(technique.module),
                technique.algorithm)
            self.options[technique.name] = algorithm
            return algorithm

    def _get_parameters(self, technique: Technique, parameters: dict):
        """Calls appropriate methods for constructing technique parameters.

        Args:
            technique (SimpleTechnique): object containing configuration 
                information for parameters to be constructed.

        Returns:
            dict containing parameters for the technique. Data dependent
                parameters are not incorporated at this stage.

        """
        for parameter_type in self.parameter_types:
            parameters = (
                getattr(self, ''.join(['_get_', parameter_type, 's']))(
                    technique = technique,
                    parameters = parameters))
        return parameters

    def _get_ideas(self, technique: SimpleTechnique, parameters: dict):
        """Acquires parameters from Idea instance.

        If the 'parameters' argument already has parameters, no changes are
        made. The parameters from the Idea instance are only incorporated if
        no existing parameters are set.

        Args:
            technique (SimpleTechnique): object containing configuration 
                information for parameters to be constructed.
            parameters (dict): parameters to be modified and returned.

        Returns:
            parameters (dict): with any appropriate changes made.

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
        """Limits parameters to those appropriate to the technique.

        If 'technique.selected' is True, the keys from 'technique.defaults' are
        used to select the final returned parameters.

        If 'technique.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        Args:
            technique (SimpleTechnique): object containing configuration 
                information for parameters to be constructed.
            parameters (dict): parameters to be modified and returned.

        Returns:
            parameters (dict): with any appropriate changes made.

        """
        if technique.selected:
            if isinstance(technique.selected, list):
                parameters_to_use = technique.selected
            else:
                parameters_to_use = list(technique.defaults.keys())
            new_parameters = {}
            for key, value in parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            parameters = new_parameters
        return parameters

    def _get_extras(self, technique: SimpleTechnique, parameters: dict):
        """Adds extra parameters (mandatory additions) to 'parameters'.

        Args:
            technique (SimpleTechnique): object containing configuration 
                information for parameters to be constructed.
            parameters (dict): parameters to be modified and returned.

        Returns:
            parameters (dict): with any appropriate changes made.

        """
        try:
            parameters.update(technique.extras)
        except TypeError:
            pass
        return parameters

    def _get_runtimes(self, technique: SimpleTechnique, parameters: dict):
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        The runtime variables should be stored as attributes in the subclass so
        that the values listed in technique.runtimes match those attributes to
        be added to parameters.

        Args:
            technique (SimpleTechnique): object containing configuration 
                information for parameters to be constructed.
            parameters (dict): parameters to be modified and returned.

        Returns:
            parameters (dict): with any appropriate changes made.

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
        """Modifies 'parameters' based upon various conditions.

        A subclass should have its own '_get_conditionals' method for this
        method to modify 'parameters'. This method is a mere placeholder.

        Args:
            technique (SimpleTechnique): object containing configuration 
                information for parameters to be constructed.
            parameters (dict): parameters to be modified and returned.

        Returns:
            parameters (dict): with any appropriate changes made.

        """
        return parameters

    """ Core siMpLify Methods """

    def draft(self):
        """Sets default techniques for a Composer subclass.

        If a subclass wishes to use gpu-dependent algorithms, that subclass
        should either include the code below or this method should be called via
        super().draft().

        """
        # Subclasses should create SimpleTechnique instances here.
        if self.gpu:
            self.add_gpu_techniques()
        return self

    def publish(self, technique: str, parameters: dict = None):
        """Converts Simpletechnique to a SimpleAlgorithm.

        Args:
            technique (str): name of technique to be used. It should match the
                name of a local attribute in the subclass.
            parameters (dict, optional): parameters to be modified and returned.
                Defaults to None.

        Returns:
            SimpleAlgorithm instance based upon the passed 'technique' and
                'parameters' (if applicable).

        """
        if technique in ['none', 'None', None]:
            return None
        else:
            # Changes technique from str to matching SimpleTechnique instance.
            technique = getattr(self, technique)
            # Acquires algorithm based upon 'technique' settings.
            algorithm = self._get_algorithm(technique = technique)
            # Determines parameters based upon technique settings.
            parameters = self._get_parameters(
                technique = technique,
                parameters = parameters)
            # Returns a SimpleAlgorithm instance.
            return SimpleAlgorithm(
                technique = technique.name,
                algorithm = algorithm,
                parameters = parameters,
                data_dependents = technique.data_dependents)

    """ Properties """

    @property
    def all(self):
        return list(self.options.keys())

    @property
    def defaults(self):
        try:
            return self._defaults
        except AttributeError:
            return list(self.options.keys())

    @defaults.setter
    def defaults(self, techniques):
        self._defaults = techniques
        
    @property
    def options(self):
        """Returns dictionary of attribute names and values if they are
        subclasses of SimpleTechnique.

        This property is used instead of a maintained 'options' dictionary in
        order to allow users to add SimpleTechnique instances to the subclasses
        in a variety of ways.

        """
        return {k: v for (k, v) in self.__dict__.items() if (issubclass(v,
                    SimpleTechnique) or isinstance(v, SimpleTechnique))}


@dataclass
class SimpleTechnique(object):
    """Stores settings to import and create a SimpleAlgorithm.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        module (str): name of internal or external module which contains the
            'algorithm' object.
        algorithm (str): name of the object to be imported from 'module'.
        defaults (dict): default parameters.
        extras (dict): extra (mandatory) parameters to be added.
        runtimes (dict): parameters to be added that are only available at
            runtime. The keys are the keys from the final intended parameters.
            The values are the names of attributes from the SimpleComposer
            subclass.
        data_dependents (dict): parameters to be added that are directly derived
            from the first passed argument to the SimpleAlgorithm 'publish'
            method. The keys are the keys from the final intended parameters.
            The values are the names of attributes from the first passed
            argument to the SimpleAlgorithm 'publish' method.
        selected (bool or list): if True, parameters will be limited to the keys
            of 'defaults'. If a list, only the matching parameter names will
            be included.

        """

    name: str = 'generic_technique'
    module: str = None
    algorithm: str = None
    defaults: object = None
    extras: object = None
    runtimes: object = None
    data_dependents: object = None
    selected: bool = False


@dataclass
class SimpleAlgorithm(SimpleClass):
    """Wraps or contains an algorithm to be applied with passed parameters.

    Args:
        technique (str): the public name of the selected algorithm.
        algorithm (object): class or function containing an algorithm to be
            applied to the variable(s) passed to the 'publish' method.
        parameters (dict): corresponding parameters for the passed algorithm.
        data_dependents (dict): parameters that are derived from the variables
            passed to the 'publish' method. The keys are names of the
            parameters and the values are the attributes to the first passed
            variable to 'publish'.
    """

    technique: str
    algorithm: object
    parameters: object
    data_dependents: object = None

    def __post_init__(self):
        # super() is not called to limit the memory used by a subclass.
        self.draft()
        return self

    """ Private Methods """

    def _add_parameters(self):
        """Attaches class instance 'parameters' to the class instance
        'algorithm'.

        """
        try:
            self.algorithm = self.algorithm(**self.parameters)
        except AttributeError:
            try:
                self.algorithm = self.algorithm(self.parameters)
        return self

    def _add_data_dependents(self, variable: object):
        """Adds data-derived parameters to the class instance parameters.

        Args:
            variable (object): class that contains attributes matching the
                values in the 'data_dependents' attribute.

        """
        for key, value in self.data_dependents.items():
            self.parameters.update({key, getattr(variable, value)})
        self._add_parameters()
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """Attaches 'parameters' to 'algorithm' if there are no data-derived
        parameters.

        """
        if not self.data_dependents:
            self._add_parameterss()
        return self

    def publish(self, variable: object, other: object = None, *args,
                **kwargs):
        """Subclasses should provide their own 'publish' methods.

        Args:
            variable (object): the class which provides the source information/
                data for the SimpleAlgorithm subclass to utilize.
            other (object): a secondary source of data/information or limits to
                be placed on the use of 'variable'.
            *args, **kwargs (parameters): any other parameters to be passed to
                called methods in the subclass.

        If a subclass wishes to use data-dependent parameters, that subclass
        should either include the code below or this method should be called via
        super().publish(variable, other).

        """
        if self.data_dependents:
            self._add_data_dependents()
        return self

