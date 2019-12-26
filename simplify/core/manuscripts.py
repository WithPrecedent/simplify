"""
.. module:: book
:synopsis: subpackage base classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.base import Resource
from simplify.core.base import SimpleOptions
from simplify.core.utilities import listify
from simplify.core.utilities import numpy_shield
from simplify.core.utilities import XxYy


@dataclass
class Book(SimpleOptions):
    """Stores and iterates Chapters.

    Args:
        project ('Project'): current associated project.

    Args:
        project ('Project'): associated Project instance.
        options (Optional[Dict[str, 'Resource']]): SimpleOptions instance or
            a SimpleOptions-compatible dictionary. Defaults to an empty
            dictionary.
        steps (Optional[Union[List[str], str]]): steps of key(s) to iterate in
            'options'. Also, if not reset by the user, 'steps' is used if the
            'default' property is accessed. Defaults to an empty list.

    """
    project: 'Project' = None
    options: Optional[Dict[str, 'Resource']] = field(default_factory = dict)
    steps: Optional[Union['SimpleSequence', List[str], str]] = field(
        default_factory = list)
    name: Optional[str] = None
    chapter_type: Optional['Chapter'] = None
    iterable: Optional[str] = 'chapters'
    metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Calls parent method for initialization.
        super().__post_init__()
        return self

    """ Core SiMpLify Methods """

    def apply(self,
            options: Optional[Union[List[str], Dict[str, Any], str]] = None,
            data: Optional[Union['Ingredients', 'Book']] = None,
            **kwargs) -> Union['Ingredients', 'Book']:
        """Calls 'apply' method for published option matching 'step'.

        Args:
            options (Optional[Union[List[str], Dict[str, Any], str]]): ordered
                options to be applied. If none are passed, the 'published' keys
                are used. Defaults to None
            data (Optional[Union['Ingredients', 'Book']]): a siMpLify object for
                the corresponding 'options' to apply. Defaults to None.
            kwargs: any additional parameters to pass to the options' 'apply'
                method.

        Returns:
            Union['Ingredients', 'Book'] is returned if data is passed;
                otherwise nothing is returned.

        """
        if isinstance(options, dict):
            options = list(options.keys())
        elif options is None:
            options = self.default
        self._change_active(new_active = 'applied')
        for option in options:
            if data is None:
                getattr(self, self.active)[option].apply(**kwargs)
            else:
                data = getattr(self, self.active)[option].apply(
                    data = data,
                    **kwargs)
            getattr(self, self.active)[option] = getattr(
                self, self.active)[option]
        if data is None:
            return self
        else:
            return data


@dataclass
class Chapter(SimpleOptions):
    """Iterator for a siMpLify process.

    Args:
        book ('Book'): current associated Book
        metadata (Optional[Dict[str, Any]], optional): any metadata about the
            Chapter. Unless a subclass replaces it, 'number' is automatically a
            key created for 'metadata' to allow for better recordkeeping.
            Defaults to an empty dictionary.

    """
    book: 'Book' = None
    name: Optional[str] = None
    iterable: Optional[str] = 'book.steps'
    metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'chapter'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _apply_extra_processing(self) -> None:
        """Extra actions to take."""
        return self

    """ Core siMpLify Methods """

    def apply(self, data: Optional['Ingredients'] = None, **kwargs) -> None:
        """Applies stored 'options' to passed 'data'.

        Args:
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): a
                siMpLify object for the corresponding 'step' to apply. Defaults
                to None.
            kwargs: any additional parameters to pass to the step's 'apply'
                method.

        """
        if data is not None:
            self.ingredients = data
        for step in getattr(self, self.iterable):
            self.book[step].apply(data = self.ingredients, **kwargs)
            self._apply_extra_processing()
        return self


@dataclass
class Page(SimpleOptions):
    """Stores, combines, and applies Algorithm and Parameters instances.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    book: 'Book' = None
    name: Optional[str] = None
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'chapter'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def _add_parameters_to_algorithm(self):
        """Attaches 'parameters' to the 'algorithm'."""
        try:
            self.algorithm = self.algorithm(**self.parameters)
        except AttributeError:
            try:
                self.algorithm = self.algorithm(self.parameters)
            except AttributeError:
                pass
        except TypeError:
            pass
        return self

    """ Public Methods """

    def draft(self):
        """Creates 'algorithm' and 'outline' attributes."""
        # Injects attributes from Idea instance, if values exist.
        self = self.options.idea.apply(instance = self)
        self.outline = self.options[self.technique]
        self.algorithm = self.outline.load()
        return self

    def publish(self) -> None:
        """Finalizes 'algorithm' and 'parameters' attributes."""
        self.algorithm = self.algorithm.publish()
        self.parameters = self.parameters.publish()
        return self

    def apply(self, data: object, **kwargs) -> object:
        """

        """
        if 'data_dependent' in self.outline:
            self.parameters._build_data_dependent(data = data)
        self._add_parameters_to_algorithm()
        try:
            self.algorithm.fit(
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])))
            setattr(
                data, ''.join(['x_', data.state]),
                self.algorithm.transform(getattr(
                    data, ''.join(['x_', data.state]))))
        except AttributeError:
            try:
                data = self.algorithm.apply(data = data)
            except AttributeError:
                pass
        return data

    """ Scikit-Learn Compatibility Methods """

    @XxYy(truncate = True)
    # @numpy_shield
    def fit(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> None:
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Raises:
            AttributeError if no 'fit' method exists for local 'algorithm'.

        """
        if x is not None:
            try:
                if y is None:
                    self.algorithm.process.fit(x)
                else:
                    self.algorithm.process.fit(x, y)
            except AttributeError:
                error = ' '.join([self.design.name,
                                  'algorithm has no fit method'])
                raise AttributeError(error)
        elif data is not None:
            self.algorithm.process.fit(
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])))
        else:
            error = ' '.join([self.name, 'algorithm has no fit method'])
            raise AttributeError(error)
        return self

    @XxYy(truncate = True)
    # @numpy_shield
    def fit_transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> (
                Union[pd.DataFrame, 'Ingredients']):
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or ingredients is not passed to
                the method.

        """
        self.algorithm.process.fit(x = x, y = y, data = ingredients)
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            return self.algorithm.process.transform(x = x, y = y)
        elif data is not None:
            return self.algorithm.process.transform(data = ingredients)
        else:
            error = ' '.join([self.name,
                              'algorithm has no fit_transform method'])
            raise TypeError(error)

    @XxYy(truncate = True)
    # @numpy_shield
    def transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> (
                Union[pd.DataFrame, 'Ingredients']):
        """Generic transform method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'process'.

        """
        if hasattr(self.algorithm.process, 'transform'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    return self.algorithm.process.transform(x)
                else:
                    return self.algorithm.process.transform(x, y)
            elif data is not None:
                return self.algorithm.process.transform(
                    X = getattr(data, 'x_' + data.state),
                    Y = getattr(data, 'y_' + data.state))
        else:
            error = ' '.join([self.name, 'algorithm has no transform method'])
            raise AttributeError(error)


@dataclass
class Algorithm(object):
    """Base class for building an algorithm for a Page subclass instance.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        _parent (Optional['Page']): optional way to set 'parent' property.

    """
    name: Optional[str] = None
    _parent: Optional['Page'] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def apply(self, outline: 'Option', **kwargs) -> object:
        """Builds and returns an algorithm.

        Args:
            outline (Optional['Option']): instance containing information
                needed to build an algorithm.
            kwargs: ignored by this class.

        Returns:
            object: a loaded algorithm.

        """
        return self._lazily_load_algorithm(outline = outline)



@dataclass
class Parameters(SimpleOptions):
    """Collection of parameters with methods for automatic construction.

    Args:
        parameters (Optional[Dict[str, Any]]):
        defaults (Optional[Union[List[str], str]]): key(s) to use if the
            'default' property is accessed.
        related (Optional['SimpleManuscript']):

    """
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)
    defaults: Optional[Union[List[str], str]] = field(default_factory = list)
    related: Optional['SimpleManuscript'] = None

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        self.active = 'parameters'
        super().__post_init__()
        return self

    """ Private Methods """

    def _publish_idea(self) -> None:
        """Acquires parameters from Idea instance, if no parameters exist."""
        if not self.parameters:
            try:
                getattr(self, self.active).update(self.idea['_'.join([
                    self.related.name, 'parameters'])])
            except AttributeError:
                pass
        return self

    def _publish_selected(self) -> None:
        """Limits parameters to those appropriate to the outline.

        If 'outline.selected' is True, the keys from 'outline.defaults' are
        used to select the final returned parameters.

        If 'outline.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        """
        if self.outline.selected:
            if isinstance(self.outline.selected, list):
                parameters_to_use = self.outline.selected
            else:
                parameters_to_use = list(self.outline.default.keys())
            new_parameters = {}
            for key, value in getattr(self, self.active).items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            setattr(self, self.active, new_parameters)
        return self

    def _publish_required(self) -> None:
        """Adds required parameters (mandatory additions) to 'parameters'."""
        try:
            getattr(self, self.active).update(self.outline.required)
        except TypeError:
            pass
        return self

    def _publish_search(self) -> None:
        """Separates variables with multiple options to search parameters."""
        self.space = {}
        if self.outline.hyperparameter_search:
            new_parameters = {}
            for parameter, values in getattr(self, self.active).items():
                if isinstance(values, list):
                    if any(isinstance(i, float) for i in values):
                        self.space.update(
                            {parameter: uniform(values[0], values[1])})
                    elif any(isinstance(i, int) for i in values):
                        self.space.update(
                            {parameter: randint(values[0], values[1])})
                else:
                    new_parameters.update({parameter: values})
            setattr(self, self.active, new_parameters)
        return self

    def _publish_runtime(self) -> None:
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        The runtime variables should be stored as attributes in the SimpleManuscript
        instance so that the values listed in outline.runtimes match those
        attributes to be added to parameters.

        """
        try:
            for key, value in self.outline.runtime.items():
                try:
                    getattr(self, self.active).update(
                        {key: getattr(self.related, value)})
                except AttributeError:
                    raise AttributeError(' '.join(
                        ['no matching runtime parameter', key, 'found in',
                         self.related.name]))
        except (AttributeError, TypeError):
            pass
        return self

    def _publish_conditional(self) -> None:
        """Modifies 'parameters' based upon various conditions.

        A related class should have its own '_publish_conditional' method for
        this method to modify 'published'. That method should have a
        'parameters' and 'name' (str) argument and return the modified
        'parameters'.

        """
        if 'conditional' in self.outline:
            try:
                setattr(self, self.active, self.related._publish_conditional(
                    name = self.outline.name,
                    parameters = getattr(self, self.active)))
            except AttributeError:
                pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Declares applicable 'parameter_types'.
        self.parameter_types = [
            'idea',
            'selected',
            'required',
            # 'search',
            'runtime',
            'conditional']
        return self

    def publish(self) -> None:
        """Finalizes parameter the active dictionary."""
        self.outline = outline
        # Updates 'active' accessing appropriate stored dictionary.
        self.change_active(active = 'published', copy_previous = True)
        for parameter_type in self.parameter_types:
            getattr(self, '_'.join(['_publish', parameter_type]))()
        return self

    def apply(self, data: object) -> None:
        """Completes parameter dictionary by adding data dependent parameters.

        Args:
            data (object): data object with attributes for data dependent
                parameters to be added.

        Returns:
            parameters with any data dependent parameters added.

        """
        # Updates 'active' accessing appropriate stored dictionary.
        self.change_active(active = 'applied', copy_previous = True)
        if self.outline.data_dependents is not None:
            for key, value in self.data_dependents.items():
                try:
                    getattr(self, self.active).update(
                        {key, getattr(data, value)})
                except KeyError:
                    print('no matching parameter found for', key, 'in',
                        data.name)
        return getattr(self, self.active)


@dataclass
class Reference(object):
    """Base class for drafting and publishing Reference instances.

    Args:
        related ('Reference'): Reference instance for methods to be applied.

    """
    related: 'Reference'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        self.draft()
        return self

    def draft(self) -> None:

        return self


@dataclass
class PageOption(Resource):
    """Contains settings for creating an Algorithm and Parameters.

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
        component (str): name of python object within 'module' to load.

    """
    name: str
    module: str
    component: str
    default: Optional[Dict[str, Any]] = None
    required: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, str]] = None
    selected: Optional[Union[bool, List[str]]] = False
    conditional: Optional[bool] = False
    data_dependent: Optional[Dict[str, str]] = None

