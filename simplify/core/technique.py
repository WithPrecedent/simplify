"""
.. module:: technique
:synopsis: technique for operating on siMpLify objects
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, List, Dict, Union, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from simplify.core.base import SimpleClass
from simplify.core.decorators import numpy_shield
from simplify.core.ingredients import Ingredients


@dataclass
class SimpleComposer(SimpleClass):
    """

    Args:
        SimpleClass ([type]): [description]
    """

    technique: str
    name: str = 'simple_composer'

    def __post_init__(self) -> None:
        super().__post_init__()
        
        return self


@dataclass
class SimpleTechnique(SimpleClass):
    """

    """

    design: 'SimpleDesign'

    def __post_init__(self) -> None:
        # Adopts string name in 'design' as this class's 'name'.
        self.name = self.design.name
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __repr__(self) -> Union[object, None]:
        """Returns finalized algorithm if is finalized. Otherwise, returns None.

        Returns:
            'algorithm' (SimpleAlgorithm): finalized algorithm, if it exists.

        """
        return self.__str__()

    def __str__(self) -> Union[object, None]:
        """Returns finalized algorithm if is finalized. Otherwise, returns None.

        Returns:
            'algorithm' (SimpleAlgorithm): finalized algorithm, if it exists.

        """
        try:
            return self.algorithm
        except AttributeError:
            return None

    """ Private Methods """

    def _join_parameters(self) -> None:
        """Attaches 'parameters' to the '_algorithm'."""
        try:
            self.algorithm = self._algorithm(**self.parameters)
        except AttributeError:
            self.algorithm = self._algorithm(self.parameters)
        except TypeError:
            pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        self._algorithm = SimpleAlgorithm(design = design)
        self.parameters = SimpleParameters(design = design)
        return self

    @numpy_shield
    def publish(self,
            data: Union[Ingredients, Tuple]) -> Union[Ingredients, Tuple, None]:
        """

        """
        if not self.design in ['none', None]:
            if self.data_dependents:
                self._add_data_dependents(ingredients = ingredients)
            if self.hyperparameter_search:
                self.algorithm = self._search_hyperparameters(
                    ingredients = ingredients,
                    data_to_use = data_to_use)
            try:
                self.algorithm.fit(
                    X = getattr(ingredients, ''.join(['x_', data_to_use])),
                    Y = getattr(ingredients, ''.join(['y_', data_to_use])),
                    **kwargs)
                setattr(ingredients, ''.join(['x_', data_to_use]),
                        self.algorithm.transform(X = getattr(
                            ingredients, ''.join(['x_', data_to_use]))))
            except AttributeError:
                ingredients = self.algorithm.publish(
                    ingredients = ingredients,
                    data_to_use = data_to_use,
                    columns = columns,
                    **kwargs)
        return ingredients


    """ Scikit-Learn Compatibility Methods """

    def fit(self, x: Union[pd.DataFrame, np.ndarray] = None,
            y: Union[pd.Series, np.ndarray] = None,
            data: Ingredients = None) -> None:
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x (DataFrame or ndarray): independent variables/features.
            y (Series, or ndarray): dependent variable/label.
            data (Ingredients): instance of Ingredients containing pandas
                data objects as attributes.

        Raises:
            AttributeError if no 'fit' method exists for local 'algorithm'.

        """
        if x is not None:
            try:
                if y is None:
                    self.algorithm.fit(x)
                else:
                    self.algorithm.fit(x, y)
            except AttributeError:
                error = ' '.join([self.design.name,
                                  'algorithm has no fit method'])
                raise AttributeError(error)
        elif data is not None:
            self.algorithm.fit(
                getattr(data, 'x_' + data.state),
                getattr(data, 'y_' + data.state))
        else:
            error = ' '.join([self.design.name, 'algorithm has no fit method'])
            raise AttributeError(error)
        return self

    def fit_transform(self, x: Union[pd.DataFrame, np.ndarray] = None,
            y: Union[pd.Series, np.ndarray] = None,
            data: Ingredients = None) -> Union[pd.DataFrame, Ingredients]:
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x (DataFrame or ndarray): independent variables/features.
            y (Series, or ndarray): dependent variable/label.
            data (Ingredients): instance of Ingredients containing pandas
                data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or ingredients is not passed to
                the method.

        """
        self.fit(x = x, y = y, ingredients = ingredients)
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            return self.transform(x = x, y = y)
        elif ingredients is not None:
            return self.transform(ingredients = ingredients)
        else:
            error = 'fit_transform requires DataFrame, ndarray, or Ingredients'
            raise TypeError(error)

    def transform(self, x: Union[pd.DataFrame, np.ndarray] = None,
            y: Union[pd.Series, np.ndarray] = None,
            data: Ingredients = None) -> Union[pd.DataFrame, Ingredients]:
        """Generic transform method for partial compatibility to sklearn.

        Args:
            x (DataFrame or ndarray): independent variables/features.
            y (Series, or ndarray): dependent variable/label.
            data (Ingredients): instance of Ingredients containing pandas
                data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'algorithm'.

        """
        if hasattr(self.algorithm, 'transform'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    return self.algorithm.transform(x)
                else:
                    return self.algorithm.transform(x, y)
            elif ingredients is not None:
                return self.algorithm.transform(
                    X = getattr(ingredients, 'x_' + self.data_to_train),
                    Y = getattr(ingredients, 'y_' + self.data_to_train))
        else:
            error = ('transform method does not exist for '
                     + self.design + ' algorithm')
            raise AttributeError(error)


@dataclass
class SimpleDesign(object):
    """Contains settings for creating a SimpleAlgorithm."""

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


@dataclass
class SimpleAlgorithm(SimpleClass):
    """Finalizes an algorithm with parameters.

    Args:
        design (SimpleDesign):

    """
    design: 'SimpleDesign'

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __repr__(self) -> Union[object, None]:
        """Returns '_gizmo' if it exists. Otherwise, returns None.

        Returns:
            '_gizmo' (object): algorithm, if it exists.

        """
        return self.__str__()

    def __str__(self) -> Union[object, None]:
        """Returns '_gizmo' if it exists. Otherwise, returns None.

        Returns:
            '_gizmo' (object): algorithm, if it exists.

        """
        try:
            return self._gizmo
        except AttributeError:
            return None

    """ Private Methods """

    def _get_algorithm(self):
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

    def _search_hyperparameter(self, ingredients: Ingredients,
                               data_to_use: str):
        search = SearchComposer()
        search.space = self.space
        search.estimator = self.algorithm
        return search.publish(ingredients = ingredients)

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



@dataclass
class SimpleParameters(SimpleClass):
    """Creates and stores parameter sets for SimpleDesigns.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'parameters_builder'
    initial_parameters: Dict = None

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __repr__(self) -> Dict:
        return self.__str__()

    def __str__(self) -> Union[Dict, None]:
        """Returns parameter dict if is finalized. Otherwise, returns None.

        Returns:
            'bunch' attribute (dict) or None, if it doesn't exist.

        """
        try:
            return self.bunch
        except AttributeError:
            return None

    """ Private Methods """

    def _get_idea(self) -> None:
        """Acquires parameters from Idea instance.

        The parameters from the Idea instance are only incorporated if no
        'initial_parameters' are set.

        """
        if self.initial_parameters is None:
            self.bunch = {}
            try:
                self.bunch.update(
                    self.idea['_'.join([design.name, 'parameters'])])
            except KeyError:
                try:
                    self.bunch.update(
                        self.idea['_'.join([design.step, 'parameters'])])
                except KeyError:
                    pass
        else:
             self.bunch = self.initial_parameters
        return self

    def _get_selected(self) -> None:
        """Limits parameters to those appropriate to the design.

        If 'design.selected' is True, the keys from 'design.defaults' are
        used to select the final returned parameters.

        If 'design.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        """
        if design.selected:
            if isinstance(design.selected, list):
                parameters_to_use = design.selected
            else:
                parameters_to_use = list(design.defaults.keys())
            new_parameters = {}
            for key, value in self.bunch.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            self.bunch = new_parameters
        return self

    def _get_required(self) -> None:
        """Adds required parameters (mandatory additions) to 'parameters'."""
        try:
            self.bunch.update(design.required)
        except TypeError:
            pass
        return self

    def _get_search(self) -> None:
        """Separates variables with multiple options to search parameters."""
        self.space = {}
        if design.hyperparameter_search:
            new_parameters = {}
            for parameter, values in self.bunch.items():
                if isinstance(values, list):
                    if any(isinstance(i, float) for i in values):
                        self.space.update(
                            {parameter: uniform(values[0], values[1])})
                    elif any(isinstance(i, int) for i in values):
                        self.space.update(
                            {parameter: randint(values[0], values[1])})
                else:
                    new_parameters.update({parameter: values})
            self.bunch = new_parameters
        return self

    def _get_runtime(self) -> None:
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        The runtime variables should be stored as attributes in the subclass so
        that the values listed in design.runtimes match those attributes to
        be added to parameters.

        """
        try:
            for key, value in design.runtimes.items():
                try:
                    self.bunch.update({key: getattr(self.step, value)})
                except AttributeError:
                    error = ' '.join('no matching runtime parameter',
                                     key, 'found in', self.step.name)
                    raise AttributeError(error)
        except TypeError:
            pass
        return self

    def _get_conditional(self) -> None:
        """Modifies 'parameters' based upon various conditions.

        A step class should have its own '_get_conditional' method for this
        method to modify 'parameters'. That method should have a 'parameters'
        argument and return the modified 'parameters'.

        """
        if self.design.conditional:
            try:
                self.bunch = self.step._get_conditional(parameters = self.bunch)
            except AttributeError:
                pass
        return self

    def _get_data_dependent(self) -> None:
        """Adds data-derived parameters to parameters 'bunch'."""
        try:
            for key, value in design.data_dependents.items():
                self.bunch.update({key, getattr(self.data, value)})
        except (KeyError, AttributeError):
            pass
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """Declares parameter_types."""
        self.parameter_types = [
            'idea',
            'selected',
            'required',
            'search',
            'runtime',
            'conditional',
            'data_dependent']
        return self

    def publish(self, step: 'SimpleClass' = None,
                data: 'SimpleClass' = None) -> None:
        """Finalizes parameter 'bunch'.

        Args:
            step (SimpleClass): step which contains a '_get_condtional' method,
                if applicable.
            data (SimpleClass): data container (Ingredients, Review, etc.) that
                has attributes matching any items stored in
                'design.data_dependent'.

        """
        if step is not None:
            self.step = step
        if data is not None:
            self.data = data
        for parameter_type in self.parameter_types:
            parameters = (
                getattr(self, '_'.join(['_get', parameter_type]))(
                    design = design,
                    parameters = parameters))
        return self