"""
.. module:: critic
:synopsis: model evaluation made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.repository import Repository
from simplify.core.repository import Sequence
from simplify.core.technique import TechniqueOutline


@dataclass
class Collection(Book):
    """Applies techniques to 'Cookbook' instances to assess performance.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'collection'.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'reviews'.
        techiques (Optional['Repository']): a dictionary of options with
            'Technique' instances stored by step. Defaults to an empty
            'Repository' instance.
        chapters (Optional['Sequence']): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty 'Sequence'
            instance.
        returns_data (Optional[bool]): whether the Scholar instance's 'apply'
            expects data when the Book instance is iterated. If False, nothing
            is returned. If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = 'collection'
    iterable: Optional[str] = 'reviews'
    techniques: Optional['Repository'] = field(default_factory = Repository)
    chapters: Optional['Sequence'] = field(default_factory = Sequence)
    returns_data: Optional[bool] = True


@dataclass
class Evaluators(Repository):
    """A dictonary of TechniqueOutline options for the Chef subpackage.

    Args:
        contents (Optional[str, Any]): default stored dictionary. Defaults to
            an empty dictionary.
        wildcards (Optional[List[str]]): a list of corresponding properties
            which access sets of dictionary keys. If none is passed, the two
            included properties ('default' and 'all') are used.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        null_value (Optional[Any]): value to return when 'none' is accessed or
            an item isn't found in 'contents'. Defaults to None.
        project ('Project'): a related 'Project' instance.

    """
    contents: Optional[Dict[str, Any]] = field(default_factory = dict)
    wildcards: Optional[List[str]] = field(default_factory = list)
    defaults: Optional[List[str]] = field(default_factory = list)
    null_value: Optional[Any] = None
    project: 'Project' = None

    """ Private Methods """

    def _create_contents(self) -> None:
        self.contents = {
            'explain': {
                'eli5': TechniqueOutline(
                    name = 'eli5_explain',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'Eli5Explain'),
                'shap': TechniqueOutline(
                    name = 'shap_explain',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'ShapExplain'),
                'skater': TechniqueOutline(
                    name = 'skater_explain',
                    module = 'skater',
                    algorithm = '')},
            'predict': {
                'gini': TechniqueOutline(
                    name = 'gini_predict',
                    module = None,
                    algorithm = 'predict'),
                'shap': TechniqueOutline(
                    name = 'shap_predict',
                    module = 'shap',
                    algorithm = '')},
            'probability': {
                'gini': TechniqueOutline(
                    name = 'gini_probabilities',
                    module = None,
                    algorithm = 'predict_proba'),
                'log': TechniqueOutline(
                    name = 'gini_probabilities',
                    module = None,
                    algorithm = 'predict_log_proba'),
                'shap': TechniqueOutline(
                    name = 'shap_probabilities',
                    module = 'shap',
                    algorithm = '')},
            'ranking': {
                'permutation': TechniqueOutline(
                    name = 'permutation_importances',
                    module = None,
                    algorithm = ''),
                'gini': TechniqueOutline(
                    name = 'gini_importances',
                    module = None,
                    algorithm = 'feature_importances_'),
                'eli5': TechniqueOutline(
                    name = 'eli5_importances',
                    module = 'eli5',
                    algorithm = ''),
                'shap': TechniqueOutline(
                    name = 'shap_importances',
                    module = 'shap',
                    algorithm = '')},
            'measure': {
                'simplify': TechniqueOutline(
                    name = 'simplify_metrics',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'compute_metrics'),
                'pandas': TechniqueOutline(
                    name = 'pandas_describe',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'pandas_describe')},
            'report': {
                'simplify': TechniqueOutline(
                    name = 'simplify_report',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'simplify_report'),
                'confusion': TechniqueOutline(
                    name = 'confusion_matrix',
                    module = 'sklearn.metrics',
                    algorithm = 'confusion_matrix'),
                'classification': TechniqueOutline(
                    name = 'classification_report',
                    module = 'sklearn.metrics',
                    algorithm = 'classification_report')}}
        return self
