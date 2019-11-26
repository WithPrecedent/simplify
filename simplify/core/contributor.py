"""
.. module:: contributor
:synopsis: algorithm and parameter builders
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from simplify.core.book import Page
from simplify.core.utilities import listify


@dataclass
class SimpleContributor(ABC):
    """Constructs pages from Outline instances for use in a Chapter.

    This class is a complex builder which constructs finalized algorithms with
    matching parameters. Because of the variance of supported packages and the
    nature of parameters involved (particularly data-dependent ones), the final
    construction of a Page is not usually completed until the 'publish' method
    of a SimpleContributor is called.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        idea ('Idea'): an instance of Idea with user settings.
        library ('Library'): an instance of Library with information about
            folder and file management.

    """
    idea: 'Idea'
    library: 'Library'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        self = self.idea.apply(instance = self)
        self.draft()
        return self

    """ Private Methods """

    def _build_page(self, page: str, ingredients: 'Ingredients') -> 'Page':
        """Builds 'page' settings in 'options'.

        Returns:
            algorithm object configured appropriately.

        """
        if page == 'none':
            return Page(algorithm = None, name = 'none')
        else:
            design = self.options[page]
            algorithm = self.algorithm_builder.publish(
                design = design)
            parameters = self.parameters_builder.publish(
                design = design,
                data = ingredients)
            return Page(
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

    def draft(self) -> None:
        """Creates builder instances."""
        self.parameters_builder = Parameters(
            idea = self.idea,
            library = self.library)
        self.algorithm_builder = Algorithm(
            idea = self.idea,
            library = self.library)
        return self

    def publish(self, page: str, ingredients: 'Ingredients') -> 'Page':
        """
        Args:
            page (str): name of page for appropriate methods and
                parameters to be returned. This name should match the name of
                a Outline class stored in 'options'.
            ingredients ('Ingredients'): instance of Ingredients containing
                data.

        """
        return self._build_page(page = page, data = ingredients)


@dataclass
class Outline(object):
    """Contains settings for creating a Algorithm and Parameters."""

    name: Optional[str] = 'simple_design'
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

@dataclass
class Algorithm(SimpleContributor):
    """Finalizes an algorithm with parameters.


    """


    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Dunder Methods """

    # def __repr__(self) -> Union[object, None]:
    #     """Returns '_gizmo' if it exists. Otherwise, returns None.

    #     Returns:
    #         '_gizmo' (object): algorithm, if it exists.

    #     """
    #     return self.__str__()

    # def __str__(self) -> Union[object, None]:
    #     """Returns '_gizmo' if it exists. Otherwise, returns None.

    #     Returns:
    #         '_gizmo' (object): algorithm, if it exists.

    #     """
    #     try:
    #         return self._gizmo
    #     except AttributeError:
    #         return None

    """ Core siMpLify Methods """

    def draft(self) -> None:
        return self

    def publish(self, design: 'Outline') -> None:
        """Finalizes parameter 'bunch'.

        Args:

        """
        self.process = getattr(import_module(design.module), design.algorithm)
        return self


@dataclass
class Parameters(SimpleContributor):
    """Creates and stores parameter sets for Outlines.

    Args:
        idea ('Idea'): an instance of Idea with user settings.
        library ('Library'): an instance of Library with information about
            folder and file management.

    Attributes:
        bunch (dict): actual parameters dict. Returned by '__str__' and
            '__repr__' methods.

    """

    idea: 'Idea'
    library: 'Library'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Dunder Methods """

    # def __repr__(self) -> Dict:
    #     return self.__str__()

    # def __str__(self) -> Union[Dict, None]:
    #     """Returns parameter dict if is finalized. Otherwise, returns None.

    #     Returns:
    #         'bunch' attribute (dict) or None, if it doesn't exist.

    #     """
    #     try:
    #         return self.bunch
    #     except AttributeError:
    #         return None

    """ Private Methods """

    def _get_idea(self, design: 'Outline') -> None:
        """Acquires parameters from Idea instance.

        Args:
            design (Outline): settings for parameters to be built.

        """
        if (not hasattr(self, 'initial_parameters')
                or not self.initial_parameters):
            self.bunch = {}
            try:
                self.bunch.update(self.idea_parameters)
            except AttributeError:
                pass
        else:
             self.bunch = self.initial_parameters
        return self

    def _get_selected(self, design: 'Outline') -> None:
        """Limits parameters to those appropriate to the design.

        If 'design.selected' is True, the keys from 'design.defaults' are
        used to select the final returned parameters.

        If 'design.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        Args:
            design (Outline): settings for parameters to be built.

        """
        if design.selected:
            if isinstance(design.selected, list):
                parameters_to_use = design.selected
            else:
                parameters_to_use = list(design.default.keys())
            new_parameters = {}
            for key, value in self.bunch.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            self.bunch = new_parameters
        return self

    def _get_required(self, design: 'Outline') -> None:
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            design (Outline): settings for parameters to be built.

        """
        try:
            self.bunch.update(design.required)
        except TypeError:
            pass
        return self

    def _get_search(self, design: 'Outline') -> None:
        """Separates variables with multiple options to search parameters.

        Args:
            design (Outline): settings for parameters to be built.

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

    def _get_runtime(self,
            design: 'Outline',
            page: 'Page') -> None:
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        The runtime variables should be stored as attributes in the subclass so
        that the values listed in design.runtimes match those attributes to
        be added to parameters.

        Args:
            design (Outline): settings for parameters to be built.

        """
        try:
            for key, value in design.runtime.items():
                try:
                    self.bunch.update({key: getattr(page, value)})
                except AttributeError:
                    error = ' '.join('no matching runtime parameter',
                                     key, 'found in', page.name)
                    raise AttributeError(error)
        except (AttributeError, TypeError):
            pass
        return self

    def _get_conditional(self,
            design: 'Outline',
            page: 'Page') -> None:
        """Modifies 'parameters' based upon various conditions.

        A page class should have its own '_get_conditional' method for this
        method to modify 'parameters'. That method should have a 'parameters'
        and 'page' (str) argument and return the modified 'parameters'.

        Args:
            design (Outline): settings for parameters to be built.

        """
        if design.conditional:
            try:
                self.bunch = page._get_conditional(
                    page = design.name,
                    parameters = self.bunch)
            except AttributeError:
                pass
        return self

    def _get_data_dependent(self,
            design: 'Outline',
            ingredients: 'Ingredients') -> None:
        """Adds data-derived parameters to parameters 'bunch'.

        Args:
            design (Outline): settings for parameters to be built.

        """
        try:
            for key, value in design.data_dependents.items():
                self.bunch.update({key, getattr(ingredients, value)})
        except (KeyError, AttributeError):
            pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
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

    def publish(self,
            design: 'Outline',
            ingredients: Optional['Ingredients'] = None,
            page: Optional['Page'] = None) -> None:
        """Finalizes parameter 'bunch'.

        Args:
            page ('Page'): page which contains a '_get_condtional'
                method, if applicable.
            ingredients ('Ingredients'): data container (Ingredients,
                Review, etc.)
                that has attributes matching any items stored in
                'outline.data_dependent'.

        """
        for parameter_type in self.parameter_types:
            if parameter_type in ['conditional', 'runtime']:
                getattr(self, '_'.join(['_get', parameter_type]))(
                    design = design,
                    page = page)
            elif parameter_type == 'data_dependent':
                getattr(self, '_'.join(['_get', parameter_type]))(
                    design = design,
                    data = ingredients)
            else:
                getattr(self, '_'.join(['_get', parameter_type]))(
                    design = design)
        return self