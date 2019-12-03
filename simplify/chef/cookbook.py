"""
.. module:: cookbook
:synopsis: data preprocesing and modeling
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.typesetter import SimpleDirector
from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.book import Page


@dataclass
class Cookbook(Book):
    """Creates recipes for staging, machine learning, and data analysis.

    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supoorted file type with settings for an Idea instance is
            located.
        library (Optional[Union['Library', str]]): an instance of
            library or a string containing the full path of where the root
            folder should be located for file output. A library instance
            contains all file path and import/export methods for use throughout
            the siMpLify package. Default is None.
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Library instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance. Default is None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        steps (Optional[Union[List[str], str]]): ordered names of pages to be
            included. These names should match keys in the 'options' attribute.
            If using the Idea instance settings, this argument should not be
            passed. Default is None.
        recipes (Optional[List['Recipe']]): Ordinarily, 'recipes' is not passed
            when Cookbook is instanced, but the argument is included if the
            user wishes to reexamine past recipes or manually create new
            recipes.

    """
    idea: Union['Idea', str]
    library: Optional[Union['Library', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    name: Optional[str] = 'chef'
    steps: Optional[Dict[str, 'Page']] = None
    recipes: Optional[List['Recipe']] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_options(self) -> None:
        self.options = {
            'scaler': ('simplify.chef.steps.scaler', 'Scaler'),
            'splitter': ('simplify.chef.steps.splitter', 'Splitter'),
            'encoder': ('simplify.chef.steps.encoder', 'Encoder'),
            'mixer': ('simplify.chef.steps.mixer', 'Mixer'),
            'cleaver': ('simplify.chef.steps.cleaver', 'Cleaver'),
            'sampler': ('simplify.chef.steps.sampler', 'Sampler'),
            'reducer': ('simplify.chef.steps.reducer', 'Reducer'),
            'modeler': ('simplify.chef.steps.modeler', 'Modeler')}
        return self

    def _extra_processing(self,
            chapter: 'Chapter',
            ingredients: 'Ingredients') -> Tuple['Chapter', 'Ingredients']:
        """Extra actions to take for each recipe."""
        if self.export_results:
            self.library.set_book_folder()
            self.library.set_chapter_folder(
                chapter = chapter,
                name = 'recipe')
            if self.export_all_recipes:
                self.save_recipes(recipes = chapter)
            if 'reduce' in self.steps and chapter.steps['reduce'] != 'none':
                ingredients.save_dropped(folder = self.library.recipe)
            else:
                ingredients.save_dropped(folder = self.library.book)
        return chapter, ingredients

    """ Public Tool Methods """

    def add_cleaves(self,
            cleave_group: str,
            prefixes: Union[List[str], str] = None,
            columns: Union[List[str], str] = None) -> None:
        """Adds cleaves to the list of cleaves.

        Args:
            cleave_group (str): names the set of features in the group.
            prefixes (Union[List[str], str]): name(s) of prefixes to columns to
                be included within the cleave.
            columns (Union[List[str], str]): name(s) of columns to be included
                within the cleave.

        """
        # if not self._exists('cleaves'):
        #     self.cleaves = []
        # columns = self.ingredients.create_column_list(
        #     prefixes = prefixes,
        #     columns = columns)
        # self.options['cleaver'].add_pages(
        #     cleave_group = cleave_group,
        #     columns = columns)
        # self.cleaves.append(cleave_group)
        return self

    """ Public Import/Export Methods """

    def load_recipe(self, file_path: str) -> None:
        """Imports a single recipe from disk and adds it to the class iterable.

        Args:
            file_path: a path where the file to be loaded is located.

        """
        self.load_chapter(file_path = file_path)
        return self

    def save_recipes(self,
            recipes: Optional[Union[List['Chapter'], str]] = None,
            file_path: Optional[str] = None) -> None:
        """Exports a recipe or recipes to disk.

        Args:
            recipe (Optional[Union[List['Chapter'], str]]): an instance of
                Recipe, a list of Recipe instances, 'all' (meaning all recipes
                stored in the class iterable), or 'best' (meaning the current
                best recipe).
            file_path (Optional[str]): path of where file should be saved. If
                None, a default file_path will be created from the shared
                Library instance.

        """
        if recipes in ['all'] or isinstance(recipes, list):
            if recipes in ['all']:
                recipes = self.recipes
            for recipe in recipes:
                self.library.set_chapter_folder(chapter = recipe)
                recipe.save(folder = self.library.recipe)
        # elif recipes in ['best'] and hasattr(self, 'critic'):
        #     self.critic.best_recipe.save(
        #         file_path = file_path,
        #         folder = self.library.book,
        #         file_name = 'best_recipe')
        elif not isinstance(recipes, str):
            recipes.save(
                file_path = file_path,
                folder = self.library.chapter)
        return

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets default options for the Chef's cookbook."""
        self.parent_type = 'project'
        self.children_type = 'recipes'
        super().draft()
        return self

    # def publish(self, ingredients: 'Ingredients') -> None:
    #     """Completes an iteration of a Cookbook.

    #     Args:
    #         ingredients (Optional['Ingredients']): If passed, it will be
    #             assigned to the local 'ingredients' attribute. If not passed,
    #             the local 'ingredients' will be used.

    #     """
    #     self.data = ingredients
    #     if 'train_test_val' in self.data_to_use:
    #         self.ingredients.state = 'train_test'
    #         super().publish(data = self.ingredients)
    #         self.ingredients.state = 'train_val'
    #         super().publish(data = self.ingredients)
    #     elif 'full' in self._data_to_use:
    #         self.ingredients.state = 'full'
    #         super().publish(data = self.ingredients)
    #     else:
    #         self.ingredients.state = 'train_test'
    #         super().publish(data = self.ingredients)
    #     if self.export_results:
    #         self.save_recipes(recipes = 'best')
    #     return self


@dataclass
class Recipe(Chapter):
    """Contains steps for analyzing data in the siMpLify Cookbook subpackage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        number (int): number of chapter in a sequence - used for recordkeeping
            purposes.
        steps (dict(str: str)): keys are names of steps and values are
            algorithms to be applied.

    """
    pages: Dict[str, 'Page']
    name: Optional[str] = 'recipe'
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        self.proxies = {'parent': 'recipes'}
        super().__post_init__()
        return self

    """ Private Methods """

    def _calculate_hyperparameters(self) -> None:
        """Computes hyperparameters that can be determined by the source data
        (without creating data leakage problems).

        This method currently only support xgboost's scale_pos_weight
        parameter. Future hyperparameter computations will be added as they
        are discovered.

        """
        if self.steps['model'] in ['xgboost']:
            # Model class is injected with scale_pos_weight for algorithms that
            # use that parameter.
            self.model.scale_pos_weight = (
                    len(self.ingredients.y.index) /
                    ((self.ingredients.y == 1).sum())) - 1
        return self


@dataclass
class RecipeStep(Page):
    """Stores, combines, and applies Algorithm and Parameters instances.

    A SimpleDirector directs the building of the requisite algorithm and
    parameters to be injected into a Page instance. When possible, these Page
    instances are made to be scikit-learn compatible using the included
    'fit', 'transform', and 'fit_transform' methods. A Page instance can also
    be applied to data using the normal siMpLify 'apply' method.

    Args:
        components (Dict[str, object])
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.

    """
    components: Dict[str, object]
    name: str = 'page'
    file_format: str = 'pickle'
    export_folder: str = 'chapter'

    def __post_init__(self) -> None:
        self.proxies = {'book': 'chapter'}
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Attaches 'parameters' to the 'algorithm'.

        """
        try:
            self.algorithm = self.algorithm.process(**self.parameters)
        except AttributeError:
            try:
                self.algorithm = self.algorithm.process(self.parameters)
            except AttributeError:
                pass
        except TypeError:
            pass
        return self

    # @numpy_shield
    def publish(self, ingredients: 'Ingredients') -> 'Ingredients':
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
            data: Optional['Ingredients'] = None) -> None:
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
            data: Optional['Ingredients'] = None) -> (
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
            data: Optional['Ingredients'] = None) -> (
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
class RecipeOutline(object):
    """Contains settings for creating a Algorithm and Parameters."""

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