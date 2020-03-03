"""
.. module:: analyst
:synopsis: machine learning made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
from inspect import signature
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from simplify.analyst import algorithms
from simplify.core.base import SimpleSettings
from simplify.core.creators import Publisher
from simplify.core.library import Book
from simplify.core.library import Chapter
from simplify.core.library import Technique
from simplify.core.manager import Worker
from simplify.core.repository import Repository
from simplify.core.scholar import Finisher
from simplify.core.scholar import Parallelizer
from simplify.core.scholar import Scholar
from simplify.core.scholar import Specialist
from simplify.core.utilities import listify


""" Book Subclasses """

@dataclass
class Cookbook(Book):
    """Standard class for iterable storage in the Analyst subpackage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'cookbook'
        chapters (Optional[List['Chapter']]): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty list.
        iterable(Optional[str]): name of property to store alternative proxy
            to 'recipes'.

    """
    name: Optional[str] = field(default_factory = lambda: 'cookbook')
    chapters: Optional[List['Chapter']] = field(default_factory = list)
    iterable: Optional[str] = field(default_factory = lambda: 'recipes')


@dataclass
class Recipe(Chapter):
    """Standard class for bottom-level Analyst subpackage iterable storage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.
        steps (Optional[List[Tuple[str, str]]]): tuples of steps and
            techniques.
        techniques (Optional[List['Technique']]): 'Technique' instances to
            apply. In an ordinary project, 'techniques' are not passed to a
            Chapter instance, but are instead created from 'steps' when the
            'publish' method of a 'Project' instance is called. Defaults to
            an empty list.

    """
    name: Optional[str] = None
    steps: Optional[List[Tuple[str, str]]] = field(default_factory = list)
    techniques: Optional[List['Technique']] = field(default_factory = list)

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable of 'techniques' or 'steps'.

        Returns:
            Iterable: of 'techniques' or 'steps', if 'techniques' do not exist.

        """
        if self.techniques:
            return iter(self.techniques)
        else:
            return iter(self.steps)

    def __len__(self) -> int:
        """Returns length of 'techniques' or 'steps'.

        Returns:
            Integer: length of 'techniques' or 'steps', if 'techniques' do not
                exist.

        """
        if self.techniques:
            return len(self.techniques)
        else:
            return len(self.steps)

    """ Proxy Property Methods """

    def _proxy_getter(self) -> List['Technique']:
        """Proxy getter for 'techniques'.

        Returns:
            List['Technique'].

        """
        return self.techniques

    def _proxy_setter(self, value: List['Technique']) -> None:
        """Proxy setter for 'techniques'.

        Args:
            value (List['Technique']): list of 'Technique' instances to store.

        """
        self.techniques = value
        return self

    def _proxy_deleter(self) -> None:
        """Proxy deleter for 'techniques'."""
        self.techniques = []
        return self

    """ Public Methods """

    def add(self,
            techniques: Union[
                List['Technique'],
                'Technique',
                List[Tuple[str, str]],
                Tuple[str, str]]) -> None:
        """Combines 'techniques' with 'steps' or 'techniques' attribute.

        If a tuple or list of tuples is passed, 'techniques' are added to the
        'steps' attribute. Otherwise, they are added to the 'techniques'
        attribute.

        Args:
            techniques (Union[List['Technique'], 'Technique', List[Tuple[str,
                str]], Tuple[str, str]]): a 'Technique' instance or tuple used
                to create one.

        """
        if isinstance(listify(techniques)[0], Tuple):
            self.steps.extend(listify(techniques))
        else:
            self.techniques.extend(listify(techniques))
        return self


""" Technique Subclass and Decorator """

def numpy_shield(callable: Callable) -> Callable:
    """
    """
    @wraps(callable)
    def wrapper(*args, **kwargs):
        call_signature = signature(callable)
        arguments = dict(call_signature.bind(*args, **kwargs).arguments)
        try:
            x_columns = list(arguments['x'].columns.values)
            result = callable(*args, **kwargs)
            if isinstance(result, np.ndarray):
                result = pd.DataFrame(result, columns = x_columns)
        except KeyError:
            result = callable(*args, **kwargs)
        return result
    return wrapper


@dataclass
class Tool(Technique):
    """Base method wrapper for applying algorithms to data.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        step (Optional[str]): name of step when the class instance is to be
            applied. Defaults to None.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        algorithm (Optional[object]): callable object which executes the primary
            method of a class instance. Defaults to None.
        parameters (Optional[Dict[str, Any]]): parameters to be attached to
            'algorithm' when 'algorithm' is instanced. Defaults to an empty
            dictionary.

    """
    name: Optional[str] = None
    step: Optional[str] = None
    module: Optional[str] = None
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)
    parameter_space: Optional[Dict[str, List[Union[int, float]]]] = field(
        default_factory = dict)
    fit_method: Optional[str] = field(default_factory = lambda: 'fit')
    transform_method: Optional[str] = field(
        default_factory = lambda: 'transform')

    """ Core siMpLify Methods """

    def apply(self, data: 'Dataset') -> 'Dataset':
        if data.stages.current in ['full']:
            self.fit(x = data.x, y = data.y)
            data.x = self.transform(x = data.x, y = data.y)
        else:

            self.fit(x = data.x_train, y = data.y_train)
            data.x_train = self.transform(x = data.x_train, y = data.y_train)
            data.x_test = self.transform(x = data.x_test, y = data.y_test)
        return data

    """ Scikit-Learn Compatibility Methods """

    @numpy_shield
    def fit(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.

        Raises:
            AttributeError if no 'fit' method exists for 'technique'.

        """
        x, y = check_X_y(X = x, y = y, accept_sparse = True)
        if self.fit_method is not None:
            if y is None:
                getattr(self.algorithm, self.fit_method)(x)
            else:
                self.algorithm = self.algorithm.fit(x, y)
        return self

    @numpy_shield
    def transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """Generic transform method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'process'.

        """
        if self.transform_method is not None:
            try:
                return getattr(self.algorithm, self.transform_method)(x)
            except AttributeError:
                return x
        else:
            return x


""" Publisher Subclass """

@dataclass
class AnalystPublisher(Publisher):
    """Creates 'Cookbook'

    Args:
        idea ('Idea'): an 'Idea' instance with project settings.

    """

    """ Public Methods """

    # def add_cleaves(self,
    #         cleave_group: str,
    #         prefixes: Union[List[str], str] = None,
    #         columns: Union[List[str], str] = None) -> None:
    #     """Adds cleaves to the list of cleaves.

    #     Args:
    #         cleave_group (str): names the set of features in the group.
    #         prefixes (Union[List[str], str]): name(s) of prefixes to columns to
    #             be included within the cleave.
    #         columns (Union[List[str], str]): name(s) of columns to be included
    #             within the cleave.

    #     """
    #     # if not self._exists('cleaves'):
    #     #     self.cleaves = []
    #     # columns = self.dataset.make_column_list(
    #     #     prefixes = prefixes,
    #     #     columns = columns)
    #     # self.workers['cleaver'].add_techniques(
    #     #     cleave_group = cleave_group,
    #     #     columns = columns)
    #     # self.cleaves.append(cleave_group)
    #     return self


""" Scholar Subclasses """

@dataclass
class AnalystScholar(Scholar):
    """Applies a 'Cookbook' instance to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self = self.idea.apply(instance = self)
        # Creates 'Finisher' instance to finalize 'Technique' instances.
        self.finisher = AnalystFinisher(worker = self.worker)
        # Creates 'Specialist' instance to apply 'Technique' instances.
        self.specialist = AnalystSpecialist(worker = self.worker)
        # Creates 'Parallelizer' instance to apply 'Chapter' instances, if the
        # option to parallelize has been selected.
        if self.parallelize:
            self.parallelizer = Parallelizer(idea = self.idea)
        return self

    """ Private Methods """

    def _get_model_type(self, data: 'Dataset') -> str:
        """Infers 'model_type' from data type of 'label' column.

        Args:
            data ('Dataset'): instance with completed dataset.

        Returns:
            str: containing the name of one of the supported model types.

        Raises:
            TypeError: if 'label' attribute is neither None, 'boolean',
                'category', 'integer' or 'float' data type (using siMpLify
                proxy datatypes).

        """
        if self.label is None:
            return 'clusterer'
        elif data.datatypes[self.label] in ['boolean']:
            return 'classifier'
        elif data.datatypes[self.label] in ['category']:
            if len(data[self.label.value_counts()]) == 2:
                return 'classifier'
            else:
                return 'multi_classifier'
        elif data.datatypes[self.label] in ['integer', 'float']:
            return 'regressor'
        else:
            raise TypeError(
                'label must be boolean, category, integer, float, or None')


@dataclass
class AnalystFinisher(Finisher):
    """Finalizes 'Technique' instances with data-dependent parameters.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional['Idea'] = None

    """ Private Methods """

    def _add_model_conditionals(self,
            technique: 'Technique',
            data: 'Dataset') -> 'Technique':
        """Adds any conditional parameters to 'technique'

        Args:
            technique ('Technique'): an instance with 'algorithm' and
                'parameters' not yet combined.
            data ('Dataset'): data object used to derive hyperparameters.

        Returns:
            'Technique': with any applicable parameters added.

        """
        self._model_calculate_hyperparameters(
            technique = technique,
            data = data)
        if technique.name in ['xgboost'] and self.idea['general']['gpu']:
            technique.parameters['tree_method'] = 'gpu_exact'
        elif step in ['tensorflow']:
            technique.algorithm = algorithms.make_tensorflow_model(
                technique = technique,
                data = data)
        return technique

    def _model_calculate_hyperparameters(self,
            technique: 'Technique',
            data: 'Dataset') -> 'Technique':
        """Computes hyperparameters from data.

        This method will include any heuristics or methods for creating smart
        algorithm parameters (without creating data leakage problems).

        This method currently only support xgboost's scale_pos_weight
        parameter. Future hyperparameter computations will be added as they
        are discovered.

        Args:
            technique ('Technique'): an instance with 'algorithm' and
                'parameters' not yet combined.
            data ('Dataset'): data object used to derive hyperparameters.

        Returns:
            'Technique': with any applicable parameters added.

        """
        if (technique.name in ['xgboost']
                and self.idea['analyst']['calculate_hyperparameters']):
            technique.parameters['scale_pos_weight'] = (
                len(self.data.y.index) / ((self.data.y == 1).sum())) - 1
        return self


@dataclass
class AnalystSpecialist(Specialist):
    """Base class for applying 'Technique' instances to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional['Idea'] = None

    """ Private Methods """

    def _apply_techniques(self,
            manuscript: 'Chapter',
            data: 'Dataset') -> 'Chapter':
        """Applies a 'chapter' of 'steps' to 'data'.

        Args:
            chapter ('Chapter'): instance with 'steps' to apply to 'data'.
            data (Union['Dataset', 'Book']): object for 'chapter' to be applied.

        Return:
            'Chapter': with any changes made. Modified 'data' is added to the
                'Chapter' instance with the attribute name matching the 'name'
                attribute of 'data'.

        """
        data.create_xy()
        for i, technique in enumerate(manuscript.techniques):
            if self.verbose:
                print('Applying', technique.name, 'to', data.name)
            if technique.step in ['split']:
                manuscript, data = self._split_loop(
                    chapter = manuscript,
                    index = i,
                    data = data)
                break
            elif technique.step in ['search']:
                remaining = self._search_loop(
                    steps = remaining,
                    index = i,
                    data = data)
                data = technique.apply(data = data)
            elif not technique.name in ['none', None]:
                data = technique.apply(data = data)
        setattr(manuscript, 'data', data)
        return manuscript

    def _split_loop(self,
            chapter: 'Chapter',
            index: int,
            data: 'DataSet') -> ('Chapter', 'Dataset'):
        """Splits 'data' and applies remaining steps in 'chapter'.

        Args:
            chapter ('Chapter'): instance with 'steps' to apply to 'data'.
            index (int): number of step in 'chapter' 'steps' where split method
                is located. All subsequent steps are completed with data split
                into training and testing sets.
            data ('Dataset'): data object for 'chapter' to be applied.

        Return:
            'Chapter', 'Dataset': with any changes made.

        """
        data.stages.change('testing')
        split_algorithm = chapter.techniques[index].algorithm
        for i, (train_index, test_index) in enumerate(
            split_algorithm.split(data.x, data.y)):
            if self.verbose:
                print('Testing data fold', str(i))
            data.x_train = data.x.iloc[train_index]
            data.x_test = data.x.iloc[test_index]
            data.y_train = data.y[train_index]
            data.y_test = data.y[test_index]
            for technique in chapter.techniques[index + 1:]:
                if self.verbose:
                    print('Applying', technique.name, 'to', data.name)
                if not technique.name in ['none', None]:
                    data = technique.apply(data = data)
        return chapter, data

    def _search_loop(self,
            chapter: 'Chapter',
            index: int,
            data: 'DataSet') -> ('Chapter', 'Dataset'):
        """Searches hyperparameters for a particular 'algorithm'.

        Args:
            chapter ('Chapter'): instance with 'steps' to apply to 'data'.
            index (int): number of step in 'chapter' 'steps' where the search
                method should be applied
            data ('Dataset'): data object for 'chapter' to be applied.

        Return:
            'Chapter': with the searched step modified with the best found
                hyperparameters.

        """
        return chapter



""" Options """

@dataclass
class Tools(Repository, SimpleSettings):
    """A dictonary of Tool options for the Analyst subpackage.

    Args:
        idea (Optional['Idea']): shared 'Idea' instance with project settings.

    """
    idea: Optional['Idea'] = None

    def create(self) -> None:
        self.contents = {
            'fill': {
                'defaults': Tool(
                    name = 'defaults',
                    module = 'simplify.analyst.algorithms',
                    algorithm = 'smart_fill',
                    default = {'defaults': {
                        'boolean': False,
                        'float': 0.0,
                        'integer': 0,
                        'string': '',
                        'categorical': '',
                        'list': [],
                        'datetime': 1/1/1900,
                        'timedelta': 0}}),
                'impute': Tool(
                    name = 'defaults',
                    module = 'sklearn.impute',
                    algorithm = 'SimpleImputer',
                    default = {'defaults': {}}),
                'knn_impute': Tool(
                    name = 'defaults',
                    module = 'sklearn.impute',
                    algorithm = 'KNNImputer',
                    default = {'defaults': {}})},
            'categorize': {
                'automatic': Tool(
                    name = 'automatic',
                    module = 'simplify.analyst.algorithms',
                    algorithm = 'auto_categorize',
                    default = {'threshold': 10}),
                'binary': Tool(
                    name = 'binary',
                    module = 'sklearn.preprocessing',
                    algorithm = 'Binarizer',
                    default = {'threshold': 0.5}),
                'bins': Tool(
                    name = 'bins',
                    module = 'sklearn.preprocessing',
                    algorithm = 'KBinsDiscretizer',
                    default = {
                        'strategy': 'uniform',
                        'n_bins': 5},
                    selected = True,
                    required = {'encode': 'onehot'})},
            'scale': {
                'gauss': Tool(
                    name = 'gauss',
                    module = None,
                    algorithm = 'Gaussify',
                    default = {'standardize': False, 'copy': False},
                    selected = True,
                    required = {'rescaler': 'standard'}),
                'maxabs': Tool(
                    name = 'maxabs',
                    module = 'sklearn.preprocessing',
                    algorithm = 'MaxAbsScaler',
                    default = {'copy': False},
                    selected = True),
                'minmax': Tool(
                    name = 'minmax',
                    module = 'sklearn.preprocessing',
                    algorithm = 'MinMaxScaler',
                    default = {'copy': False},
                    selected = True),
                'normalize': Tool(
                    name = 'normalize',
                    module = 'sklearn.preprocessing',
                    algorithm = 'Normalizer',
                    default = {'copy': False},
                    selected = True),
                'quantile': Tool(
                    name = 'quantile',
                    module = 'sklearn.preprocessing',
                    algorithm = 'QuantileTransformer',
                    default = {'copy': False},
                    selected = True),
                'robust': Tool(
                    name = 'robust',
                    module = 'sklearn.preprocessing',
                    algorithm = 'RobustScaler',
                    default = {'copy': False},
                    selected = True),
                'standard': Tool(
                    name = 'standard',
                    module = 'sklearn.preprocessing',
                    algorithm = 'StandardScaler',
                    default = {'copy': False},
                    selected = True)},
            'split': {
                'group_kfold': Tool(
                    name = 'group_kfold',
                    module = 'sklearn.model_selection',
                    algorithm = 'GroupKFold',
                    default = {'n_splits': 5},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    fit_method = None,
                    transform_method = 'split'),
                'kfold': Tool(
                    name = 'kfold',
                    module = 'sklearn.model_selection',
                    algorithm = 'KFold',
                    default = {'n_splits': 5, 'shuffle': False},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    required = {'shuffle': True},
                    fit_method = None,
                    transform_method = 'split'),
                'stratified': Tool(
                    name = 'stratified',
                    module = 'sklearn.model_selection',
                    algorithm = 'StratifiedKFold',
                    default = {'n_splits': 5, 'shuffle': False},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    required = {'shuffle': True},
                    fit_method = None,
                    transform_method = 'split'),
                'time': Tool(
                    name = 'time',
                    module = 'sklearn.model_selection',
                    algorithm = 'TimeSeriesSplit',
                    default = {'n_splits': 5},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    fit_method = None,
                    transform_method = 'split'),
                'train_test': Tool(
                    name = 'train_test',
                    module = 'sklearn.model_selection',
                    algorithm = 'ShuffleSplit',
                    default = {'test_size': 0.33},
                    runtime = {'random_state': 'seed'},
                    required = {'n_splits': 1},
                    selected = True,
                    fit_method = None,
                    transform_method = 'split')},
            'encode': {
                'backward': Tool(
                    name = 'backward',
                    module = 'category_encoders',
                    algorithm = 'BackwardDifferenceEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'basen': Tool(
                    name = 'basen',
                    module = 'category_encoders',
                    algorithm = 'BaseNEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'binary': Tool(
                    name = 'binary',
                    module = 'category_encoders',
                    algorithm = 'BinaryEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'dummy': Tool(
                    name = 'dummy',
                    module = 'category_encoders',
                    algorithm = 'OneHotEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'hashing': Tool(
                    name = 'hashing',
                    module = 'category_encoders',
                    algorithm = 'HashingEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'helmert': Tool(
                    name = 'helmert',
                    module = 'category_encoders',
                    algorithm = 'HelmertEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'james_stein': Tool(
                    name = 'james_stein',
                    module = 'category_encoders',
                    algorithm = 'JamesSteinEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'loo': Tool(
                    name = 'loo',
                    module = 'category_encoders',
                    algorithm = 'LeaveOneOutEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'm_estimate': Tool(
                    name = 'm_estimate',
                    module = 'category_encoders',
                    algorithm = 'MEstimateEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'ordinal': Tool(
                    name = 'ordinal',
                    module = 'category_encoders',
                    algorithm = 'OrdinalEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'polynomial': Tool(
                    name = 'polynomial_encoder',
                    module = 'category_encoders',
                    algorithm = 'PolynomialEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'sum': Tool(
                    name = 'sum',
                    module = 'category_encoders',
                    algorithm = 'SumEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'target': Tool(
                    name = 'target',
                    module = 'category_encoders',
                    algorithm = 'TargetEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'woe': Tool(
                    name = 'weight_of_evidence',
                    module = 'category_encoders',
                    algorithm = 'WOEEncoder',
                    data_dependent = {'cols': 'categoricals'})},
            'mix': {
                'polynomial': Tool(
                    name = 'polynomial_mixer',
                    module = 'sklearn.preprocessing',
                    algorithm = 'PolynomialFeatures',
                    default = {
                        'degree': 2,
                        'interaction_only': True,
                        'include_bias': True}),
                'quotient': Tool(
                    name = 'quotient',
                    module = None,
                    algorithm = 'QuotientFeatures'),
                'sum': Tool(
                    name = 'sum',
                    module = None,
                    algorithm = 'SumFeatures'),
                'difference': Tool(
                    name = 'difference',
                    module = None,
                    algorithm = 'DifferenceFeatures')},
            'cleave': {
                'cleaver': Tool(
                    name = 'cleaver',
                    module = 'simplify.analyst.algorithms',
                    algorithm = 'Cleaver')},
            'sample': {
                'adasyn': Tool(
                    name = 'adasyn',
                    module = 'imblearn.over_sampling',
                    algorithm = 'ADASYN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'cluster': Tool(
                    name = 'cluster',
                    module = 'imblearn.under_sampling',
                    algorithm = 'ClusterCentroids',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'knn': Tool(
                    name = 'knn',
                    module = 'imblearn.under_sampling',
                    algorithm = 'AllKNN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'near_miss': Tool(
                    name = 'near_miss',
                    module = 'imblearn.under_sampling',
                    algorithm = 'NearMiss',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'random_over': Tool(
                    name = 'random_over',
                    module = 'imblearn.over_sampling',
                    algorithm = 'RandomOverSampler',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'random_under': Tool(
                    name = 'random_under',
                    module = 'imblearn.under_sampling',
                    algorithm = 'RandomUnderSampler',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'smote': Tool(
                    name = 'smote',
                    module = 'imblearn.over_sampling',
                    algorithm = 'SMOTE',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'smotenc': Tool(
                    name = 'smotenc',
                    module = 'imblearn.over_sampling',
                    algorithm = 'SMOTENC',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    data_dependent = {
                        'categorical_features': 'categoricals_indices'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'smoteenn': Tool(
                    name = 'smoteenn',
                    module = 'imblearn.combine',
                    algorithm = 'SMOTEENN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'smotetomek': Tool(
                    name = 'smotetomek',
                    module = 'imblearn.combine',
                    algorithm = 'SMOTETomek',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample')},
            'reduce': {
                'kbest': Tool(
                    name = 'kbest',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectKBest',
                    default = {'k': 10, 'score_func': 'f_classif'},
                    selected = True),
                'fdr': Tool(
                    name = 'fdr',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFdr',
                    default = {'alpha': 0.05, 'score_func': 'f_classif'},
                    selected = True),
                'fpr': Tool(
                    name = 'fpr',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFpr',
                    default = {'alpha': 0.05, 'score_func': 'f_classif'},
                    selected = True),
                'custom': Tool(
                    name = 'custom',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFromModel',
                    default = {'threshold': 'mean'},
                    runtime = {'estimator': 'algorithm'},
                    selected = True),
                'rank': Tool(
                    name = 'rank',
                    module = 'simplify.critic.rank',
                    algorithm = 'RankSelect',
                    selected = True),
                'rfe': Tool(
                    name = 'rfe',
                    module = 'sklearn.feature_selection',
                    algorithm = 'RFE',
                    default = {'n_features_to_select': 10, 'step': 1},
                    runtime = {'estimator': 'algorithm'},
                    selected = True),
                'rfecv': Tool(
                    name = 'rfecv',
                    module = 'sklearn.feature_selection',
                    algorithm = 'RFECV',
                    default = {'n_features_to_select': 10, 'step': 1},
                    runtime = {'estimator': 'algorithm'},
                    selected = True)}}
        model_options = {
            'classify': {
                'adaboost': Tool(
                    name = 'adaboost',
                    module = 'sklearn.ensemble',
                    algorithm = 'AdaBoostClassifier',
                    transform_method = None),
                'baseline_classifier': Tool(
                    name = 'baseline_classifier',
                    module = 'sklearn.dummy',
                    algorithm = 'DummyClassifier',
                    required = {'strategy': 'most_frequent'},
                    transform_method = None),
                'logit': Tool(
                    name = 'logit',
                    module = 'sklearn.linear_model',
                    algorithm = 'LogisticRegression',
                    transform_method = None),
                'random_forest': Tool(
                    name = 'random_forest',
                    module = 'sklearn.ensemble',
                    algorithm = 'RandomForestClassifier',
                    transform_method = None),
                'svm_linear': Tool(
                    name = 'svm_linear',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'linear', 'probability': True},
                    transform_method = None),
                'svm_poly': Tool(
                    name = 'svm_poly',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'poly', 'probability': True},
                    transform_method = None),
                'svm_rbf': Tool(
                    name = 'svm_rbf',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'rbf', 'probability': True},
                    transform_method = None),
                'svm_sigmoid': Tool(
                    name = 'svm_sigmoid ',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'sigmoid', 'probability': True},
                    transform_method = None),
                'tensorflow': Tool(
                    name = 'tensorflow',
                    module = 'tensorflow',
                    algorithm = None,
                    default = {
                        'batch_size': 10,
                        'epochs': 2},
                    transform_method = None),
                'xgboost': Tool(
                    name = 'xgboost',
                    module = 'xgboost',
                    algorithm = 'XGBClassifier',
                    # data_dependent = 'scale_pos_weight',
                    transform_method = None)},
            'cluster': {
                'affinity': Tool(
                    name = 'affinity',
                    module = 'sklearn.cluster',
                    algorithm = 'AffinityPropagation',
                    transform_method = None),
                'agglomerative': Tool(
                    name = 'agglomerative',
                    module = 'sklearn.cluster',
                    algorithm = 'AgglomerativeClustering',
                    transform_method = None),
                'birch': Tool(
                    name = 'birch',
                    module = 'sklearn.cluster',
                    algorithm = 'Birch',
                    transform_method = None),
                'dbscan': Tool(
                    name = 'dbscan',
                    module = 'sklearn.cluster',
                    algorithm = 'DBSCAN',
                    transform_method = None),
                'kmeans': Tool(
                    name = 'kmeans',
                    module = 'sklearn.cluster',
                    algorithm = 'KMeans',
                    transform_method = None),
                'mean_shift': Tool(
                    name = 'mean_shift',
                    module = 'sklearn.cluster',
                    algorithm = 'MeanShift',
                    transform_method = None),
                'spectral': Tool(
                    name = 'spectral',
                    module = 'sklearn.cluster',
                    algorithm = 'SpectralClustering',
                    transform_method = None),
                'svm_linear': Tool(
                    name = 'svm_linear',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM',
                    transform_method = None),
                'svm_poly': Tool(
                    name = 'svm_poly',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM',
                    transform_method = None),
                'svm_rbf': Tool(
                    name = 'svm_rbf',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM,',
                    transform_method = None),
                'svm_sigmoid': Tool(
                    name = 'svm_sigmoid',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM',
                    transform_method = None)},
            'regress': {
                'adaboost': Tool(
                    name = 'adaboost',
                    module = 'sklearn.ensemble',
                    algorithm = 'AdaBoostRegressor',
                    transform_method = None),
                'baseline_regressor': Tool(
                    name = 'baseline_regressor',
                    module = 'sklearn.dummy',
                    algorithm = 'DummyRegressor',
                    required = {'strategy': 'mean'},
                    transform_method = None),
                'bayes_ridge': Tool(
                    name = 'bayes_ridge',
                    module = 'sklearn.linear_model',
                    algorithm = 'BayesianRidge',
                    transform_method = None),
                'lasso': Tool(
                    name = 'lasso',
                    module = 'sklearn.linear_model',
                    algorithm = 'Lasso',
                    transform_method = None),
                'lasso_lars': Tool(
                    name = 'lasso_lars',
                    module = 'sklearn.linear_model',
                    algorithm = 'LassoLars',
                    transform_method = None),
                'ols': Tool(
                    name = 'ols',
                    module = 'sklearn.linear_model',
                    algorithm = 'LinearRegression',
                    transform_method = None),
                'random_forest': Tool(
                    name = 'random_forest',
                    module = 'sklearn.ensemble',
                    algorithm = 'RandomForestRegressor',
                    transform_method = None),
                'ridge': Tool(
                    name = 'ridge',
                    module = 'sklearn.linear_model',
                    algorithm = 'Ridge',
                    transform_method = None),
                'svm_linear': Tool(
                    name = 'svm_linear',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'linear', 'probability': True},
                    transform_method = None),
                'svm_poly': Tool(
                    name = 'svm_poly',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'poly', 'probability': True},
                    transform_method = None),
                'svm_rbf': Tool(
                    name = 'svm_rbf',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'rbf', 'probability': True},
                    transform_method = None),
                'svm_sigmoid': Tool(
                    name = 'svm_sigmoid ',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'sigmoid', 'probability': True},
                    transform_method = None),
                'xgboost': Tool(
                    name = 'xgboost',
                    module = 'xgboost',
                    algorithm = 'XGBRegressor',
                    # data_dependent = 'scale_pos_weight',
                    transform_method = None)}}
        gpu_options = {
            'classify': {
                'forest_inference': Tool(
                    name = 'forest_inference',
                    module = 'cuml',
                    algorithm = 'ForestInference',
                    transform_method = None),
                'random_forest': Tool(
                    name = 'random_forest',
                    module = 'cuml',
                    algorithm = 'RandomForestClassifier',
                    transform_method = None),
                'logit': Tool(
                    name = 'logit',
                    module = 'cuml',
                    algorithm = 'LogisticRegression',
                    transform_method = None)},
            'cluster': {
                'dbscan': Tool(
                    name = 'dbscan',
                    module = 'cuml',
                    algorithm = 'DBScan',
                    transform_method = None),
                'kmeans': Tool(
                    name = 'kmeans',
                    module = 'cuml',
                    algorithm = 'KMeans',
                    transform_method = None)},
            'regressor': {
                'lasso': Tool(
                    name = 'lasso',
                    module = 'cuml',
                    algorithm = 'Lasso',
                    transform_method = None),
                'ols': Tool(
                    name = 'ols',
                    module = 'cuml',
                    algorithm = 'LinearRegression',
                    transform_method = None),
                'ridge': Tool(
                    name = 'ridge',
                    module = 'cuml',
                    algorithm = 'RidgeRegression',
                    transform_method = None)}}
        self.contents['model'] = model_options[
            self.idea['analyst']['model_type']]
        if self.idea['general']['gpu']:
            self.contents['model'].update(
                gpu_options[idea['analyst']['model_type']])
        return self.contents


@dataclass
class Analyst(Worker):
    """Object construction instructions used by a Project instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        book (Optional[str]): name of Book object in 'module' to load. Defaults
            to 'Book'.
        chapter (Optional[str]): name of Chapter object in 'module' to load.
            Defaults to 'Chapter'.
        technique (Optional[str]): name of Book object in 'module' to load.
            Defaults to 'Technique'.
        publisher (Optional[str]): name of Publisher class in 'module' to load.
            Defaults to 'Publisher'.
        scholar (Optional[str]): name of Scholar class in 'module' to load.
            Defaults to 'Scholar'.
        steps (Optional[List[str]]): list of steps to execute. Defaults to an
            empty list.
        options (Optional[Union[str, Dict[str, Any]]]): a dictionary containing
            options for the 'Worker' instance to utilize or a string
            corresponding to a dictionary in 'module' to load. Defaults to an
            empty dictionary.
        data (Optional[str]): name of attribute or key in a 'Project' instance
            'library' to use as a data object to apply methods to. Defaults to
            'dataset'.
        import_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.

    """
    name: Optional[str] = field(default_factory = lambda: 'analyst')
    module: Optional[str] = field(
        default_factory = lambda: 'simplify.analyst.analyst')
    book: Optional[str] = field(default_factory = lambda: 'Cookbook')
    chapter: Optional[str] = field(default_factory = lambda: 'Recipe')
    technique: Optional[str] = field(default_factory = lambda: 'Tool')
    publisher: Optional[str] = field(
        default_factory = lambda: 'AnalystPublisher')
    scholar: Optional[str] = field(default_factory = lambda: 'AnalystScholar')
    options: Optional[str] = field(default_factory = lambda: 'Tools')
    idea: Optional['Idea'] = None
