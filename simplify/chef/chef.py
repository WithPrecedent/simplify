"""
.. module:: chef
:synopsis: algorithm and parameter builder
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.author import Author
from simplify.core.book import Book
from simplify.core.chapter import Chapter
from simplify.core.content import Content
from simplify.core.outline import Outline
from simplify.core.page import Page
from simplify.core.utilities import listify


@dataclass
class Chef(Author):
    """Constructs Page instances from Outline instances for use in a Chapter.

    This class is a director for a complex content which constructs finalized
    algorithms with matching parameters. Because of the variance of supported
    packages and the nature of parameters involved (particularly data-dependent
    ones), the final construction of a Page is not usually completed until the
    'apply' method is called.

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
        steps (Optional[Union[List[str], str]]): ordered names of Book
            subclasses to include. These names should match keys in the
            'options' attribute. If using the Idea instance settings, this
            argument should not be passed. Default is None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'ingredients' must also be passed. Defaults to True.
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.
        export_folder (Optional[str]): attribute name of folder in 'library' for
            serialization of subclasses to be saved. Defaults to 'book'.

    """
    idea: Union['Idea', str]
    library: Optional[Union['Library', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    steps: Optional[Union[List[str], str]] = None
    name: Optional[str] = 'simplify'
    auto_publish: Optional[bool] = True
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _build_conditional(self,
            name: str,
            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Modifies 'parameters' based upon various conditions.

        A subclass should have its own '_build_conditional' method for this
        method to modify 'parameters'. That method should have a 'parameters'
        argument and return the modified 'parameters'.

        Args:
            name (str): name of method being used.
            parameters (Dict[str, Any]): a dictionary of parameters.

        Returns:
            parameters (Dict[str, Any]): altered parameters based on condtions.

        """
        pass


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
        self._options = SimpleOptions(options = {
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
class Step(SKLearnPage):


@dataclass
class Essence(Outline):
    """Contains settings for creating a Algorithm and Parameters.

    Users can use the idiom 'x in Outline' to check if a particular attribute
    exists and is not None. This means default values for optional arguments
    should generally be set to None to allow use of that idiom.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (Optional[str]): name of module where object to incorporate is
            located (can either be a siMpLify or non-siMpLify object). Defaults
            to None.
        algorithm (Optional[str]): name of object within 'module' to load.
            Defaults to None.

    """
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
