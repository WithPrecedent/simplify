"""
.. module:: technique
:synopsis: technique builder, settings, container, and application classes
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
from simplify.core.decorators import XxYy
from simplify.core.ingredients import Ingredients


@dataclass
class SimpleDesign(object):
    """Contains settings for creating a SimpleAlgorithm and SimpleParameters."""

    name: str = 'simple_design'
    step: str = ''
    module: str = None
    algorithm: str = None
    default: Dict[str, Any] = None
    required: Dict[str, Any] = None
    runtime: Dict[str, str] = None
    data_dependent: Dict[str, str] = None
    selected: Union[bool, List[str]] = False
    conditional: bool = False
    hyperparameter_search: bool = False
    

@dataclass
class SimpleComposer(SimpleClass):
    """Constructs techniques for use in SimplePlan.

    This class is a complex builder which constructs finalized algorithms with
    matching parameters. Because of the variance of supported packages and the
    nature of parameters involved (particularly data-dependent ones), the final
    construction of a technique is not usually completed until the 'publish'
    method of the technique is called.

    Args:

        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'simple_composer'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _build_technique(self, technique: str, data: SimpleClass,
                         step: SimpleClass) -> 'SimpleDesign':
        """Builds technique settings in 'options'.

        Returns:
            algorithm object configured appropriately.

        """
        design = self.options[technique]
        algorithm = self.algorithm_builder.publish(
            design = design,
            data = data)
        parameters = self.parameters_builder.publish(
            design = design,
            data = data)
        return SimpleTechnique(
            name = design.name,
            algorithm = algorithm,
            parameters = parameters)

    def _get_conditional(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
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
        self.parameters_builder = SimpleParameters()
        self.algorithm_builder = SimpleAlgorithm()
        return self

    def publish(self, technique: str, data: SimpleClass) -> 'SimpleTechnique':
        """
        Args:
            technique (str): name of technique for appropriate methods and
                parameters to be returned. This name should match the name of
                a SimpleDesign class stored in 'options'.

        """
        if technique in ['none', 'None', None]:
            return None
        else:
            # Builds technique and returns it
            return self._build_technique(
                technique = technique,
                data = data)


@dataclass
class SimpleTechnique(SimpleClass):
    """Container for SimpleAlgorithm, SimpleParameters, and finalized algorithm.

    This is the primary mechanism for storing techniques in siMpLify. The
    SimpleComposer directs the building of the requisite algorithm and
    parameters to be injected into this technique. When possible, these
    techniques are made to be scikit-learn compatible using the included
    'fit', 'transform', and 'fit_transform' methods. The techniques can also
    be applied to data using the normal siMpLify 'publish' method.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        algorithm (SimpleAlgorithm): finalized algorithm instance.
        parameters (SimpleParameters): finalized parameters instance.

    """
    name: str
    algorithm: 'SimpleAlgorithm'
    parameters: 'SimpleParameters'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __repr__(self) -> Union[object, None]:
        """Returns 'algorithm'.

        Returns:
            'algorithm' (object): finalized algorithm.

        """
        return self.__str__()

    def __str__(self) -> Union[object, None]:
        """Returns 'algorithm'.

        Returns:
            'algorithm' (object): finalized algorithm.

        """
        try:
            return self.algorithm
        except AttributeError:
            return None

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Attaches 'parameters' to the 'algorithm'.

        """
        try:
            self.algorithm = self.algorithm(**self.parameters)
        except AttributeError:
            self.algorithm = self.algorithm(self.parameters)
        return self

    @numpy_shield
    def publish(self,
            data: Union[Ingredients, Tuple]) -> Union[Ingredients, Tuple, None]:
        """

        """
        # if self.hyperparameter_search:
        #     self.algorithm = self._search_hyperparameters(
        #         ingredients = ingredients,
        #         data_to_use = data_to_use)
        try:
            self.algorithm.fit(
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])),
                **kwargs)
            setattr(
                data, ''.join(['x_', data.state]),
                self.algorithm.transform(getattr(
                    data, ''.join(['x_', data.state]))))
        except AttributeError:
            data = self.algorithm.publish(data = data, **kwargs)
        return data


    """ Scikit-Learn Compatibility Methods """

    @XxYy(truncate = True)
    @numpy_shield
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
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])))
        else:
            error = ' '.join([self.name, 'algorithm has no fit method'])
            raise AttributeError(error)
        return self

    @XxYy(truncate = True)
    @numpy_shield
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
        elif data is not None:
            return self.transform(ingredients = ingredients)
        else:
            error = ' '.join([self.name,
                              'algorithm has no fit_transform method'])
            raise TypeError(error)

    @XxYy(truncate = True)
    @numpy_shield
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
            elif data is not None:
                return self.algorithm.transform(
                    X = getattr(data, 'x_' + data.state),
                    Y = getattr(data, 'y_' + data.state))
        else:
            error = ' '.join([self.name, 'algorithm has no transform method'])
            raise AttributeError(error)


@dataclass
class SimpleAlgorithm(SimpleClass):
    """Finalizes an algorithm with parameters.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'simple_algorithm'

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

    """ Core siMpLify Methods """

    def draft(self):
        return self

    def publish(self, design: SimpleDesign) -> None:
        """Finalizes parameter 'bunch'.

        Args:

        """
        self._gizmo = getattr(import_module(design.module), design.algorithm)
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

    Attributes:
        bunch (dict): actual parameters dict. Returned by '__str__' and
            '__repr__' methods.

    """
    name: str = 'parameters_builder'

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

    def _get_idea(self, design: SimpleDesign) -> None:
        """Acquires parameters from Idea instance.

        Args:
            design (SimpleDesign): settings for parameters to be built.

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

    def _get_selected(self, design: SimpleDesign) -> None:
        """Limits parameters to those appropriate to the design.

        If 'design.selected' is True, the keys from 'design.defaults' are
        used to select the final returned parameters.

        If 'design.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        Args:
            design (SimpleDesign): settings for parameters to be built.

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

    def _get_required(self, design: SimpleDesign) -> None:
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            design (SimpleDesign): settings for parameters to be built.

        """
        try:
            self.bunch.update(design.required)
        except TypeError:
            pass
        return self

    def _get_search(self, design: SimpleDesign) -> None:
        """Separates variables with multiple options to search parameters.

        Args:
            design (SimpleDesign): settings for parameters to be built.

        """
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

    def _get_runtime(self, design: SimpleDesign,
                     step: SimpleClass) -> None:
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        The runtime variables should be stored as attributes in the subclass so
        that the values listed in design.runtimes match those attributes to
        be added to parameters.

        Args:
            design (SimpleDesign): settings for parameters to be built.

        """
        try:
            for key, value in design.runtimes.items():
                try:
                    self.bunch.update({key: getattr(step, value)})
                except AttributeError:
                    error = ' '.join('no matching runtime parameter',
                                     key, 'found in', step.name)
                    raise AttributeError(error)
        except TypeError:
            pass
        return self

    def _get_conditional(self, design: SimpleDesign, step: SimpleClass) -> None:
        """Modifies 'parameters' based upon various conditions.

        A step class should have its own '_get_conditional' method for this
        method to modify 'parameters'. That method should have a 'parameters'
        and 'technique' (str) argument and return the modified 'parameters'.

        Args:
            design (SimpleDesign): settings for parameters to be built.

        """
        if design.conditional:
            try:
                self.bunch = step._get_conditional(
                    technique = design.name,
                    parameters = self.bunch)
            except AttributeError:
                pass
        return self

    def _get_data_dependent(self, design: SimpleDesign,
                            data: SimpleClass) -> None:
        """Adds data-derived parameters to parameters 'bunch'.

        Args:
            design (SimpleDesign): settings for parameters to be built.

        """
        try:
            for key, value in design.data_dependents.items():
                self.bunch.update({key, getattr(data, value)})
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
            # 'search',
            'runtime',
            'conditional',
            'data_dependent']
        return self

    def publish(self, design: SimpleDesign, data: SimpleClass = None,
                step: SimpleClass = None) -> None:
        """Finalizes parameter 'bunch'.

        Args:
            step (SimpleClass): step which contains a '_get_condtional' method,
                if applicable.
            data (SimpleClass): data container (Ingredients, Review, etc.) that
                has attributes matching any items stored in
                'design.data_dependent'.

        """
        for parameter_type in self.parameter_types:
            if parameter_type in ['conditional', 'runtime']:
                getattr(self, '_'.join(['_get', parameter_type]))(
                    design = design,
                    step = step)
            elif parameter_type == 'data_dependent':
                getattr(self, '_'.join(['_get', parameter_type]))(
                    design = design,
                    data = data)
            else:
                getattr(self, '_'.join(['_get', parameter_type]))(
                    design = design)
        return self