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
from . import base


@dataclasses.dataclass
class Critic(base.SimpleManager):
    """Manages a distinct portion of a data science project workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a SimpleSettings
            instance, 'name' should match the appropriate section name in a 
            SimpleSettings instance. Defaults to None. 
        workflow (base.SimpleWorkflow): a workflow of a project subpart derived 
            from 'outline'. Defaults to None.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to an
            empty list.
                
    """
    name: str = 'critic'
    workflow: base.SimpleWorkflow = None
    needs: ClassVar[Union[Sequence[str], str]] = ['outline', 'name']


@dataclasses.dataclass
class Anthology(base.SimpleSummary):
    """Collects and stores results of executing a data science project workflow.
    
    Args:
        contents (Mapping[Any, Any]]): stored dictionary. Defaults to an empty 
            dict.
        default (Any): default value to return when the 'get' method is used.
        prefix (str): prefix to use when storing different paths through a 
            workflow. So, for example, a prefix of 'path' will create keys of
            'path_1', 'path_2', etc. Defaults to 'experiment'.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to 
            a list with 'workflow' and 'data'.          
              
    """
    contents: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    default: Any = None
    prefix: str = 'experiment'
    needs: ClassVar[Union[Sequence[str], str]] = ['workflow', 'data'] 
    

@dataclasses.dataclass
class SimpleEli5(base.SimpleTechnique):
    
    name: str = 'eli5_explainer'
    module: str = 'simplify.critic'
    algorithm: str = 'Eli5Explain'


@dataclasses.dataclass
class SimpleShap(base.SimpleTechnique):
    
    name: str = 'shap_explainer'
    module: str = 'simplify.critic'
    algorithm: str = 'ShapExplain'



# options = sourdough.types.Catalog(contents = {
#     'explain': {
#         'eli5': base.SimpleTechnique(
#             name = 'eli5_explain',
#             module = 'simplify.critic.explainers',
#             algorithm = 'Eli5Explain'),
#         'shap': base.SimpleTechnique(
#             name = 'shap_explain',
#             module = 'simplify.critic.explainers',
#             algorithm = 'ShapExplain'),
#         'skater': base.SimpleTechnique(
#             name = 'skater_explain',
#             module = 'simplify.critic.explainers',
#             algorithm = 'SkaterExplain'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_explain',
#             module = 'simplify.critic.explainers',
#             algorithm = 'SklearnExplain')},
#     'predict': {
#         'eli5': base.SimpleTechnique(
#             name = 'eli5_predict',
#             module = 'simplify.critic.predictors',
#             algorithm = 'Eli5Predict'),
#         'shap': base.SimpleTechnique(
#             name = 'shap_predict',
#             module = 'simplify.critic.predictors',
#             algorithm = 'ShapPredict'),
#         'skater': base.SimpleTechnique(
#             name = 'skater_predict',
#             module = 'simplify.critic.predictors',
#             algorithm = 'SkaterPredict'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_predict',
#             module = 'simplify.critic.predictors',
#             algorithm = 'SklearnPredict')},
#     'rank': {
#         'eli5': base.SimpleTechnique(
#             name = 'eli5_rank',
#             module = 'simplify.critic.rankers',
#             algorithm = 'Eli5Rank'),
#         'shap': base.SimpleTechnique(
#             name = 'shap_rank',
#             module = 'simplify.critic.rankers',
#             algorithm = 'ShapRank'),
#         'skater': base.SimpleTechnique(
#             name = 'skater_rank',
#             module = 'simplify.critic.rankers',
#             algorithm = 'SkaterRank'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_rank',
#             module = 'simplify.critic.rankers',
#             algorithm = 'SklearnRank')},
#     'measure': {
#         'simplify': base.SimpleTechnique(
#             name = 'simplify_measure',
#             module = 'simplify.critic.metrics',
#             algorithm = 'SimplifyMeasure'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_measure',
#             module = 'simplify.critic.metrics',
#             algorithm = 'SklearnMeasure')},
#     'report': {
#         'simplify': base.SimpleTechnique(
#             name = 'simplify_report',
#             module = 'simplify.critic.reporters',
#             algorithm = 'SimplifyReport'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_report',
#             module = 'simplify.critic.reporters',
#             algorithm = 'SklearnReport')}})


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



