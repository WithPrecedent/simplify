"""
.. module:: page
:synopsis: composite tree base leaf class
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from itertools import product
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

from simplify.creator.codex import SimpleCodex
from simplify.library.utilities import listify


@dataclass
class Page(SimpleCodex):
    """Stores, combines, and applies Algorithm and Parameters instances.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    algorithm: 'Algorithm'
    parameters: 'Parameters'
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.proxies = {'parent': 'chapter', 'children': 'content'}
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_content(self,
            content: Union['Content', List['Content']]) -> None:
        """Instances all 'content' and returns instanced list.

        Args:
            content (Optional[Union['Content', List['Content']]):
                instance(s) of Content subclass. Defaults to None.

        """
        instanced_content = []
        for item in listify(content):
            # Checks to see if class has already been instanced.
            if not isinstance(item, Content):
                instanced_item = item()
                instanced_item.author = self
                instanced_content.append(instanced_item)
        return instanced_content

    """ Public Methods """

    def add_content(self,
            content: Union['Content', List['Content']],
            replace_content: Optional[bool] = False) -> None:
        """Adds Content classes to 'content' attribute.

        Args:
            content (Union['Content'], List['Content']):
                subclass(es), not instance(s), of Content.
            replace_content (Optional[bool]): whether to replace existing
                'content' (True) or add them to existing 'content' (False).

        """
        if replace_content or self.content is None:
            self.content = listify(content)
        else:
            self.content.extend(listify(content))
        self.content = self._draft_content(content = self.content)
        return self

    def publish(self,
            content: Optional[Union['Content'],
                               List['Content']] = None,
            replace_content: Optional[bool] = False) -> None:
        """Validates content.

        If subclass instances provide their own methods, they should incorporate
        or call the code below.

        Args:
            content (Optional[Union['Content'], List['Content']]):
                instance(s) of Content subclass. Defaults to None.
            replace_content (Optional[bool]): if 'content' is passed, whether
                to replace existing 'content' (True) or add them to existing
                'content'.

        """
        if content is not None:
            self.add_content(
                content = content,
                replace_content = replace_content)
        for item in listify(self.content):
            item.publish()
            # Validates 'content', if possible.
            try:
                for component in item.components:
                    method = '_'.join(['_build', component])
                    if not hasattr(item, method):
                        raise NotImplementedError(' '.join([
                            content.name,
                            'requires build method for every component']))
            except AttributeError:
                pass
        return self

    def apply(self,
            page: 'Page',
            outline: Optional['Outline'],
            **kwargs) -> 'SimpleComposite':
        """Builds and returns Page object.

        If subclass instances provide their own methods, they should incorporate
        or call the code below.

        Args:
            page ('Page'): class, not instance, of page subclass to return with
                components added.
            outline (Optional['Outline']): instance containing information
                needed to build the desired objects. Defaults to None.
            kwargs (Dict[str, Any]): keyword arguments to pass to content.

        """
        if outline is None:
            outline = self.outline
        elif self.outline is None:
            self.outline = outline
        components = {}
        for component in self.content:
            components[component.name] = component.apply(
                outline = outline, **kwargs)
        return page(components = components)

    """ Core siMpLify Methods """

    def publish(self, data: Optional[object] = None) -> None:
        self.algorithm = self.algorithm.publish(data = data)
        self.parameters = self.parameters.publish(data = data)
        # Attaches 'parameters' to the 'algorithm'.
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

    def apply(self, data: object, **kwargs) -> object:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (object): data object for methods to be applied.

        Returns:
            data (object): data object with methods applied.

        """
        return self.algorithm.apply(data = data)


@dataclass
class SKLearnPage(Page):
    """
    Provides partial scikit-learn compatibility via the included 'fit',
    'transform', and 'fit_transform' methods.
    """
    """ Core siMpLify Methods """

    def publish(self) -> None:

        return self

    # @numpy_shield
    def apply(self, data: object) -> 'Ingredients':
        """[summary]

        Returns:
            [type]: [description]
        """

        # if self.hyperparameter_search:
        #     self.algorithm = self._search_hyperparameters(
        #         data = ingredients,
        #         data_to_use = data_to_use)
        try:
            self.algorithm.fit(
                getattr(ingredients, ''.join(['x_', ingredients.state])),
                getattr(ingredients, ''.join(['y_', ingredients.state])))
            setattr(
                ingredients, ''.join(['x_', ingredients.state]),
                self.algorithm.transform(getattr(
                    ingredients, ''.join(['x_', ingredients.state]))))
        except AttributeError:
            try:
                data = self.algorithm.publish(
                    data = ingredients)
            except AttributeError:
                pass
        return ingredients

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
def PageFiler(SimpleFiler):
    """
    Args:
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.
    """
    file_format: str = 'pickle'
    export_folder: str = 'chapter'
