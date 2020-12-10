"""
critic: model and feature evaluation
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
import pathlib
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import numpy as np
import pandas as pd

import simplify
import sourdough


@dataclasses.dataclass
class Criticize(simplify.SimpleProject):
    """Constructs, organizes, and implements model and feature evaluation.

    Args:
        contents (Mapping[str, object]]): stored objects created by the 
            'create' methods of 'creators'. Defaults to an empty dict.
        settings (Union[Type, str, pathlib.Path]]): a Settings-compatible class,
            a str or pathlib.Path containing the file path where a file of a 
            supported file type with settings for a Settings instance is 
            located. Defaults to the default Settings instance.
        manager (Union[Type, str, pathlib.Path]]): a Manager-compatible class,
            or a str or pathlib.Path containing the full path of where the root 
            folder should be located for file input and output. A 'manager'
            must contain all file path and import/export methods for use 
            throughout sourdough. Defaults to the default Manager instance. 
        creators (Sequence[Union[Type, str]]): a Creator-compatible classes or
            strings corresponding to the keys in registry of the default
            'creator' in 'bases'. Defaults to a list of 'simple_architect', 
            'simple_builder', and 'simple_worker'. 
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example if a 
            sourdough instance needs settings from a Settings instance, 'name' 
            should match the appropriate section name in the Settings instance. 
            When subclassing, it is sometimes a good idea to use the same 'name' 
            attribute as the base class for effective coordination between 
            sourdough classes. If it is None, the 'name' will be attempted to be 
            inferred from the first section name in 'settings' after 'general' 
            and 'files'. If that fails, 'name' will be the snakecase name of the
            class. Defaults to None. 
        identification (str): a unique identification name for a Project 
            instance. The name is used for creating file folders related to the 
            project. If it is None, a str will be created from 'name' and the 
            date and time. Defaults to None.   
        automatic (bool): whether to automatically advance 'director' (True) or 
            whether the director must be advanced manually (False). Defaults to 
            True.
        data (object): any data object for the project to be applied. If it is
            None, an instance will still execute its workflow, but it won't
            apply it to any external data. Defaults to None.  
        bases (ClassVar[object]): contains information about default base 
            classes used by a Project instance. Defaults to an instance of 
            SimpleBases.

    """
    contents: Sequence[Any] = dataclasses.field(default_factory = dict)
    settings: Union[object, Type, str, pathlib.Path] = None
    manager: Union[object, Type, str, pathlib.Path] = None
    creators: Sequence[Union[Type, str]] = dataclasses.field(
        default_factory = lambda: ['analyst_architect', 'analyst_builder', 
                                   'analyst_worker'])
    name: str = None
    identification: str = None
    automatic: bool = True
    data: Union[pd.DataFrame, np.ndArray, simplify.Dataset] = None
    bases: ClassVar[object] = simplify.SimpleBases()

    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        sourdough.rules.validations.append('data')
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        
    """ Private Methods """
    
    def _validate_data(self) -> None:
        """Validates 'data' or converts it to a Dataset instance."""
        pass


@dataclasses.dataclass
class Anthology(sourdough.Product):
    """Applies techniques to 'Cookbook' instances to assess performance.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'anthology'
        chapters (Optional[List['Chapter']]): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty list.
        iterable(Optional[str]): name of property to store alternative proxy
            to 'reviews'.
        steps (Optional[List[Tuple[str, str]]]): tuples of steps and
            techniques.
        techniques (Optional[List['simplify.SimpleTechnique']]): 'simplify.SimpleTechnique' instances to
            apply. In an ordinary project, 'techniques' are not passed to an
            Anthology instance, but are instead created from 'steps' when the
            'publish' method of a 'Project' instance is called. Defaults to
            an empty list.

    """
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'anthology')
    chapters: Optional[List['Review']] = dataclasses.field(default_factory = list)
    iterable: Optional[str] = dataclasses.field(default_factory = lambda: 'reviews')
    steps: Optional[List[Tuple[str, str]]] = dataclasses.field(default_factory = list)
    techniques: Optional[List['simplify.SimpleTechnique']] = dataclasses.field(default_factory = list)
    

options = sourdough.types.Catalog(contents = {
    'explain': {
        'eli5': simplify.SimpleTechnique(
            name = 'eli5_explain',
            module = 'simplify.critic.explainers',
            algorithm = 'Eli5Explain'),
        'shap': simplify.SimpleTechnique(
            name = 'shap_explain',
            module = 'simplify.critic.explainers',
            algorithm = 'ShapExplain'),
        'skater': simplify.SimpleTechnique(
            name = 'skater_explain',
            module = 'simplify.critic.explainers',
            algorithm = 'SkaterExplain'),
        'sklearn': simplify.SimpleTechnique(
            name = 'sklearn_explain',
            module = 'simplify.critic.explainers',
            algorithm = 'SklearnExplain')},
    'predict': {
        'eli5': simplify.SimpleTechnique(
            name = 'eli5_predict',
            module = 'simplify.critic.predictors',
            algorithm = 'Eli5Predict'),
        'shap': simplify.SimpleTechnique(
            name = 'shap_predict',
            module = 'simplify.critic.predictors',
            algorithm = 'ShapPredict'),
        'skater': simplify.SimpleTechnique(
            name = 'skater_predict',
            module = 'simplify.critic.predictors',
            algorithm = 'SkaterPredict'),
        'sklearn': simplify.SimpleTechnique(
            name = 'sklearn_predict',
            module = 'simplify.critic.predictors',
            algorithm = 'SklearnPredict')},
    'rank': {
        'eli5': simplify.SimpleTechnique(
            name = 'eli5_rank',
            module = 'simplify.critic.rankers',
            algorithm = 'Eli5Rank'),
        'shap': simplify.SimpleTechnique(
            name = 'shap_rank',
            module = 'simplify.critic.rankers',
            algorithm = 'ShapRank'),
        'skater': simplify.SimpleTechnique(
            name = 'skater_rank',
            module = 'simplify.critic.rankers',
            algorithm = 'SkaterRank'),
        'sklearn': simplify.SimpleTechnique(
            name = 'sklearn_rank',
            module = 'simplify.critic.rankers',
            algorithm = 'SklearnRank')},
    'measure': {
        'simplify': simplify.SimpleTechnique(
            name = 'simplify_measure',
            module = 'simplify.critic.metrics',
            algorithm = 'SimplifyMeasure'),
        'sklearn': simplify.SimpleTechnique(
            name = 'sklearn_measure',
            module = 'simplify.critic.metrics',
            algorithm = 'SklearnMeasure')},
    'report': {
        'simplify': simplify.SimpleTechnique(
            name = 'simplify_report',
            module = 'simplify.critic.reporters',
            algorithm = 'SimplifyReport'),
        'sklearn': simplify.SimpleTechnique(
            name = 'sklearn_report',
            module = 'simplify.critic.reporters',
            algorithm = 'SklearnReport')}})


# def _get_brier_score_loss_parameters(self, parameters, recipe = None):
#     if self.step in 'brier_score_loss':
#         parameters = {
#             'y_true': getattr(recipe.dataset,
#                                 'y_' + self.data_to_review),
#             'y_prob': recipe.probabilities[:, 1]}
#     elif self.step in ['roc_auc']:
#             parameters = {
#                 'y_true': getattr(recipe.dataset,
#                                 'y_' + self.data_to_review),
#                 'y_score': recipe.probabilities[:, 1]}
#     return parameters



