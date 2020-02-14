"""
.. module:: analyst
:synopsis: machine learning made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.book import Technique
from simplify.core.publisher import Publisher
from simplify.core.repository import Repository
from simplify.core.utilities import listify
from simplify.core.worker import Worker


""" Book Subclass """

@dataclass
class Cookbook(Book):
    """Standard class for iterable storage in the Analyst subpackage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'cookbook'
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'recipes'.
        chapters (Optional['Plan']): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty 'Plan'
            instance.

    """
    name: Optional[str] = field(default_factory = lambda: 'cookbook')
    iterable: Optional[str] = field(default_factory = lambda: 'recipes')
    chapters: Optional[List['Chapter']] = field(default_factory = list)


""" Recipe Subclass """

@dataclass
class Recipe(Chapter):

    def __post_init__(self) -> None:
        super().__post_init__()
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
class AnalystTechnique(Technique):

    """Core iterable for sequences of methods to apply to passed data.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        technique (Optional[str]): name of particular technique to be used. It
            should correspond to a key in the related 'book' instance. Defaults
            to None.

    """
    name: Optional[str] = None
    technique: Optional[str] = None
    algorithm: Optional[object] = None
    module: Optional[str]
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    data_dependents: Optional[Dict[str, str]] = field(default_factory = dict)
    parameter_space: Optional[Dict[str, List[Union[int, float]]]] = field(
        default_factory = dict)
    fit_method: Optional[str] = field(default_factory = lambda: 'fit')
    transform_method: Optional[str] = field(
        default_factory = lambda: 'transform')

    """ Required ABC Methods """

    def __contains__(self, key: str) -> bool:
        """Returns whether 'attribute' is the 'technique'.

        Args:
            key (str): name of item to check.

        Returns:
            bool: whether the 'key' is equivalent to 'technique'.

        """
        return item == self.technique

    """ Core siMpLify Methods """

    def apply(self, data: 'Dataset') -> 'Dataset':
        if data.stages.current in ['full']:
            data.x = self.fit(x = data.x, y = data.y)
        else:
            self.fit(x = data.x_train, y = data.y_train)
            data.x_train = self.transform(x = data.x_train, y = data.y_train)
            data.x_test = self.transform(x = data.x_test, y = data.y_test)
        return data

    """ Scikit-Learn Compatibility Methods """

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
        try:
            if y is None:
                getattr(self.algorithm, self.fit_method)(x)
            else:
                getattr(self.algorithm, self.fit_method)(x, y)
        except AttributeError:
            raise AttributeError(' '.join(
                [self.technique, 'has no fit method']))
        return self

    @numpy_shield
    def fit_transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or dataset is not passed to
                the method.

        """
        self.fit(x = x, y = y, data = dataset)
        return self.transform(x = x, y = y)

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
        if self.transform_method:
            return getattr(self.algorithm, self.transform_method)(x, y)
        else:
            raise AttributeError(' '.join(
                [self.technique, 'has no transform method']))


""" Publisher Subclass """

@dataclass
class AnalystPublisher(Publisher):
    """Creates 'Cookbook'

    Args:
        idea ('Idea'): an 'Idea' instance with project settings.

    """
    idea: 'Idea'

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
    #     # self.tasks['cleaver'].add_techniques(
    #     #     cleave_group = cleave_group,
    #     #     columns = columns)
    #     # self.cleaves.append(cleave_group)
    #     return self


""" Worker Subclass """

@dataclass
class Analyst(Worker):
    """Applies a 'Cookbook' instance to data.

    Args:
        idea ('Idea'): an 'Idea' instance with project settings.

    """
    idea: 'Idea'

    """ Private Methods """

    def _iterate_chapter(self,
            chapter: 'Chapter',
            data: Union['Dataset']) -> 'Chapter':
        """Iterates a single chapter and applies 'techniques' to 'data'.

        Args:
            chapter ('Chapter'): instance with 'techniques' to apply to 'data'.
            data (Union['Dataset', 'Book']): object for 'chapter'
                'techniques' to be applied.

        Return:
            'Chapter': with any changes made. Modified 'data' is added to the
                'Chapter' instance with the attribute name matching the 'name'
                attribute of 'data'.

        """
        data.create_xy()
        remaining = deepcopy(chapter.techniques)
        for step, techniques in chapter.techniques.items():
            if step in ['split']:
                data, remaining = self._split_loop(
                    steps = remaining,
                    data = data)
                break
            elif step in ['search']:
                techniques = self._search_loop(
                    techniques = techniques,
                    data = data)
            else:
                data = self._iterate_techniques(
                    techniques = techniques,
                    data = data)
            del remaining[step]
        chapter.techniques.update(remaining)
        setattr(chapter, 'data', data)
        return chapter

    def _split_loop(self,
            steps: 'Plan',
            data: 'DataSet') -> ('Dataset', 'Plan'):
        """Iterates 'steps' starting with train/test split.

        Args:

        """
        data.stages.change('testing')
        split_algorithm = steps['split'][0].algorithm
        for train_index, test_index in split_algorithm.split(data.x, data.y):
            data.x_train = data.x.iloc[train_index]
            data.x_test = data.x.iloc[test_index]
            data.y_train = data.y[train_index]
            data.y_test = data.y[test_index]
            for step, techniques in steps.items():
                print('test what are techniques', step, techniques)
                if not step in ['split'] and not techniques in ['none', None]:
                    self._iterate_techniques(
                        techniques = techniques,
                        data = data)
        return data, steps

    def _search_loop(self,
            data: 'DataSet',
            chapter: 'Chapter',
            index: int) -> 'Chapter':

        return chapter

    def _add_model_conditionals(self,
            technique: 'Technique',
            data: 'Dataset') -> 'Technique':
        """Adds any conditional parameters to 'technique'

        Args:
            technique ('Technique'): an instance with 'algorithm' and
                'parameters' not yet combined.

        Returns:
            'Technique': with any applicable parameters added.

        """
        # self._model_calculate_hyperparameters(
        #     technique = technique,
        #     data = data)
        # if technique.technique in ['xgboost'] and self.gpu:
        #     technique.parameters['tree_method'] = 'gpu_exact'
        # elif step in ['tensorflow']:
        #     technique.algorithm = make_tensorflow_model(
        #         technique = technique,
        #         data = data)
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

        Returns:
            'Technique': with any applicable parameters added.

        """
        if (technique.technique in ['xgboost']
                and self.calculate_hyperparameters):
            technique.parameters['scale_pos_weight'] = (
                    len(self.data['y'].index) /
                    ((self.data['y'] == 1).sum())) - 1
        return self

    """ Core siMpLify Methods """

    # def apply(self,
    #         book: 'Book',
    #         data: Optional[Union['Dataset', 'Book']] = None,
    #         **kwargs) -> Union['Dataset', 'Book']:
    #     """Applies objects in 'book' to 'data'.

    #     Args:
    #         book ('Book'): Book instance with algorithms to apply to 'data'.
    #         data (Optional[Union['Dataset', 'Book']]): a data source for
    #             the 'book' methods to be applied.
    #         kwargs: any additional parameters to pass to a related
    #             Book's options' 'apply' method.

    #     Returns:
    #         Union['Dataset', 'Book']: data object with modifications
    #             possibly made.

    #     """

    #     super().apply(book = book, data = data, **kwargs)
    #     return book

    # def _cleave(self, dataset):
    #     if self.step != 'all':
    #         cleave = self.tasks[self.step]
    #         drop_list = [i for i in self.test_columns if i not in cleave]
    #         for col in drop_list:
    #             if col in dataset.x_train.columns:
    #                 dataset.x_train.drop(col, axis = 'columns',
    #                                          inplace = True)
    #                 dataset.x_test.drop(col, axis = 'columns',
    #                                         inplace = True)
    #     return dataset

    # def _publish_cleaves(self):
    #     for group, columns in self.tasks.items():
    #         self.test_columns.extend(columns)
    #     if self.parameters['include_all']:
    #         self.tasks.update({'all': self.test_columns})
    #     return self

    # def add(self, cleave_group, columns):
    #     """For the cleavers in siMpLify, this step alows users to manually
    #     add a new cleave group to the cleaver dictionary.
    #     """
    #     self.tasks.update({cleave_group: columns})
    #     return self


#        self.scorers = {'f_classif': f_classif,
#                        'chi2': chi2,
#                        'mutual_class': mutual_info_classif,
#                        'mutual_regress': mutual_info_regression}

    # # @numpy_shield
    # def publish(self, dataset, plan = None, estimator = None):
    #     if not estimator:
    #         estimator = plan.model.algorithm
    #     self._set_parameters(estimator)
    #     self.algorithm = self.tasks[self.step](**self.parameters)
    #     if len(dataset.x_train.columns) > self.num_features:
    #         self.algorithm.fit(dataset.x_train, dataset.y_train)
    #         mask = ~self.algorithm.get_support()
    #         dataset.drop_columns(df = dataset.x_train, mask = mask)
    #         dataset.drop_columns(df = dataset.x_test, mask = mask)
    #     return dataset

    # # @numpy_shield
    # def publish(self,
    #         dataset: 'Dataset',
    #         data_to_use: str,
    #         columns: list = None,
    #         **kwargs) -> 'Dataset':
    #     """[summary]

    #     Args:
    #         dataset (Dataset): [description]
    #         data_to_use (str): [description]
    #         columns (list, optional): [description]. Defaults to None.
    #     """
    #     if self.step != 'none':
    #         if self.data_dependents:
    #             self._add_data_dependents(data = dataset)
    #         if self.hyperparameter_search:
    #             self.algorithm = self._search_hyperparameters(
    #                 data = dataset,
    #                 data_to_use = data_to_use)
    #         try:
    #             self.algorithm.fit(
    #                 X = getattr(dataset, ''.join(['x_', data_to_use])),
    #                 Y = getattr(dataset, ''.join(['y_', data_to_use])),
    #                 **kwargs)
    #             setattr(dataset, ''.join(['x_', data_to_use]),
    #                     self.algorithm.transform(X = getattr(
    #                         dataset, ''.join(['x_', data_to_use]))))
    #         except AttributeError:
    #             data = self.algorithm.publish(
    #                 data = dataset,
    #                 data_to_use = data_to_use,
    #                 columns = columns,
    #                 **kwargs)
    #     return dataset

    # def _set_parameters(self, estimator):
#        if self.step in ['rfe', 'rfecv']:
#            self.default = {'n_features_to_select': 10,
#                                       'step': 1}
#            self.runtime_parameters = {'estimator': estimator}
#        elif self.step == 'kbest':
#            self.default = {'k': 10,
#                                       'score_func': f_classif}
#            self.runtime_parameters = {}
#        elif self.step in ['fdr', 'fpr']:
#            self.default = {'alpha': 0.05,
#                                       'score_func': f_classif}
#            self.runtime_parameters = {}
#        elif self.step == 'custom':
#            self.default = {'threshold': 'mean'}
#            self.runtime_parameters = {'estimator': estimator}
#        self._publish_parameters()
#        self._select_parameters()
#        self.parameters.update({'estimator': estimator})
#        if 'k' in self.parameters:
#            self.num_features = self.parameters['k']
#        else:
#            self.num_features = self.parameters['n_features_to_select']
        # return self




# @dataclass
# class SearchComposer(AnalystComposer):
#     """Searches for optimal model hyperparameters using specified step.

#     Args:

#     Returns:
#         [type]: [description]
#     """
#     name: str = 'search_composer'
#     algorithm_class: object = SearchAnalystTechnique
#     step_class: object = SearchTechnique

#     def __post_init__(self) -> None:
#         self.idea_sections = ['analyst']
#         super().__post_init__()
#         return self

#     """ Private Methods """

#     def _build_conditional(self, step: AnalystTechnique, parameters: dict):
#         """[summary]

#         Args:
#             step (namedtuple): [description]
#             parameters (dict): [description]
#         """
#         if 'refit' in parameters and isinstance(parameters['scoring'], list):
#             parameters['scoring'] = parameters['scoring'][0]
#         return parameters
#         self.space = {}
#         if step.hyperparameter_search:
#             new_parameters = {}
#             for parameter, values in parameters.items():
#                 if isinstance(values, list):
#                     if self._datatype_in_list(values, float):
#                         self.space.update(
#                             {parameter: uniform(values[0], values[1])})
#                     elif self._datatype_in_list(values, int):
#                         self.space.update(
#                             {parameter: randint(values[0], values[1])})
#                 else:
#                     new_parameters.update({parameter: values})
#             parameters = new_parameters
#         return parameters

#     def _search_hyperparameter(self, dataset: Dataset,
#                                data_to_use: str):
#         search = SearchComposer()
#         search.space = self.space
#         search.estimator = self.algorithm
#         return search.publish(data = dataset)

#     """ Core siMpLify Methods """

#     def draft(self) -> None:
#         self.bayes = Technique(
#             name = 'bayes',
#             module = 'bayes_opt',
#             algorithm = 'BayesianOptimization',
#             runtime = {
#                 'f': 'estimator',
#                 'pbounds': 'space',
#                 'random_state': 'seed'})
#         self.grid = Technique(
#             name = 'grid',
#             module = 'sklearn.model_selection',
#             algorithm = 'GridSearchCV',
#             runtime = {
#                 'estimator': 'estimator',
#                 'param_distributions': 'space',
#                 'random_state': 'seed'})
#         self.random = Technique(
#             name = 'random',
#             module = 'sklearn.model_selection',
#             algorithm = 'RandomizedSearchCV',
#             runtime = {
#                 'estimator': 'estimator',
#                 'param_distributions': 'space',
#                 'random_state': 'seed'})
#         super().draft()
#         return self


# @dataclass
# class SearchAnalystTechnique(AnalystTechnique):
#     """[summary]

#     Args:
#         object ([type]): [description]
#     """
#     step: str
#     algorithm: object
#     parameters: object
#     data_dependents: object = None
#     hyperparameter_search: bool = False
#     space: object = None
#     name: str = 'search'

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     @numpy_shield
#     def publish(self, dataset: Dataset, data_to_use: str):
#         """[summary]

#         Args:
#             dataset ([type]): [description]
#             data_to_use ([type]): [description]
#         """
#         if self.step in ['random', 'grid']:
#             return self.algorithm.fit(
#                 X = getattr(dataset, ''.join(['x_', data_to_use])),
#                 Y = getattr(dataset, ''.join(['y_', data_to_use])),
#                 **kwargs)


""" Options """

@dataclass
class Tools(Repository):
    """A dictonary of AnalystTechnique options for the Analyst subpackage.

    Args:
        idea ('Idea'): shared 'Idea' instance with project settings.

    """
    idea: 'Idea'

    def create(self) -> None:
        self.contents = {
            'fill': {
                'defaults': AnalystTechnique(
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
                'impute': AnalystTechnique(
                    name = 'defaults',
                    module = 'sklearn.impute',
                    algorithm = 'SimpleImputer',
                    default = {'defaults': {}}),
                'knn_impute': AnalystTechnique(
                    name = 'defaults',
                    module = 'sklearn.impute',
                    algorithm = 'KNNImputer',
                    default = {'defaults': {}})},
            'categorize': {
                'automatic': AnalystTechnique(
                    name = 'automatic',
                    module = 'simplify.analyst.algorithms',
                    algorithm = 'auto_categorize',
                    default = {'threshold': 10}),
                'binary': AnalystTechnique(
                    name = 'binary',
                    module = 'sklearn.preprocessing',
                    algorithm = 'Binarizer',
                    default = {'threshold': 0.5}),
                'bins': AnalystTechnique(
                    name = 'bins',
                    module = 'sklearn.preprocessing',
                    algorithm = 'KBinsDiscretizer',
                    default = {
                        'strategy': 'uniform',
                        'n_bins': 5},
                    selected = True,
                    required = {'encode': 'onehot'})},
            'scale': {
                'gauss': AnalystTechnique(
                    name = 'gauss',
                    module = None,
                    algorithm = 'Gaussify',
                    default = {'standardize': False, 'copy': False},
                    selected = True,
                    required = {'rescaler': 'standard'}),
                'maxabs': AnalystTechnique(
                    name = 'maxabs',
                    module = 'sklearn.preprocessing',
                    algorithm = 'MaxAbsScaler',
                    default = {'copy': False},
                    selected = True),
                'minmax': AnalystTechnique(
                    name = 'minmax',
                    module = 'sklearn.preprocessing',
                    algorithm = 'MinMaxScaler',
                    default = {'copy': False},
                    selected = True),
                'normalize': AnalystTechnique(
                    name = 'normalize',
                    module = 'sklearn.preprocessing',
                    algorithm = 'Normalizer',
                    default = {'copy': False},
                    selected = True),
                'quantile': AnalystTechnique(
                    name = 'quantile',
                    module = 'sklearn.preprocessing',
                    algorithm = 'QuantileTransformer',
                    default = {'copy': False},
                    selected = True),
                'robust': AnalystTechnique(
                    name = 'robust',
                    module = 'sklearn.preprocessing',
                    algorithm = 'RobustScaler',
                    default = {'copy': False},
                    selected = True),
                'standard': AnalystTechnique(
                    name = 'standard',
                    module = 'sklearn.preprocessing',
                    algorithm = 'StandardScaler',
                    default = {'copy': False},
                    selected = True)},
            'split': {
                'group_kfold': AnalystTechnique(
                    name = 'group_kfold',
                    module = 'sklearn.model_selection',
                    algorithm = 'GroupKFold',
                    default = {'n_splits': 5},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    fit_method = None,
                    transform_method = 'split'),
                'kfold': AnalystTechnique(
                    name = 'kfold',
                    module = 'sklearn.model_selection',
                    algorithm = 'KFold',
                    default = {'n_splits': 5, 'shuffle': False},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    required = {'shuffle': True},
                    fit_method = None,
                    transform_method = 'split'),
                'stratified': AnalystTechnique(
                    name = 'stratified',
                    module = 'sklearn.model_selection',
                    algorithm = 'StratifiedKFold',
                    default = {'n_splits': 5, 'shuffle': False},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    required = {'shuffle': True},
                    fit_method = None,
                    transform_method = 'split'),
                'time': AnalystTechnique(
                    name = 'time',
                    module = 'sklearn.model_selection',
                    algorithm = 'TimeSeriesSplit',
                    default = {'n_splits': 5},
                    runtime = {'random_state': 'seed'},
                    selected = True,
                    fit_method = None,
                    transform_method = 'split'),
                'train_test': AnalystTechnique(
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
                'backward': AnalystTechnique(
                    name = 'backward',
                    module = 'category_encoders',
                    algorithm = 'BackwardDifferenceEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'basen': AnalystTechnique(
                    name = 'basen',
                    module = 'category_encoders',
                    algorithm = 'BaseNEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'binary': AnalystTechnique(
                    name = 'binary',
                    module = 'category_encoders',
                    algorithm = 'BinaryEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'dummy': AnalystTechnique(
                    name = 'dummy',
                    module = 'category_encoders',
                    algorithm = 'OneHotEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'hashing': AnalystTechnique(
                    name = 'hashing',
                    module = 'category_encoders',
                    algorithm = 'HashingEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'helmert': AnalystTechnique(
                    name = 'helmert',
                    module = 'category_encoders',
                    algorithm = 'HelmertEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'james_stein': AnalystTechnique(
                    name = 'james_stein',
                    module = 'category_encoders',
                    algorithm = 'JamesSteinEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'loo': AnalystTechnique(
                    name = 'loo',
                    module = 'category_encoders',
                    algorithm = 'LeaveOneOutEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'm_estimate': AnalystTechnique(
                    name = 'm_estimate',
                    module = 'category_encoders',
                    algorithm = 'MEstimateEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'ordinal': AnalystTechnique(
                    name = 'ordinal',
                    module = 'category_encoders',
                    algorithm = 'OrdinalEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'polynomial': AnalystTechnique(
                    name = 'polynomial_encoder',
                    module = 'category_encoders',
                    algorithm = 'PolynomialEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'sum': AnalystTechnique(
                    name = 'sum',
                    module = 'category_encoders',
                    algorithm = 'SumEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'target': AnalystTechnique(
                    name = 'target',
                    module = 'category_encoders',
                    algorithm = 'TargetEncoder',
                    data_dependent = {'cols': 'categoricals'}),
                'woe': AnalystTechnique(
                    name = 'weight_of_evidence',
                    module = 'category_encoders',
                    algorithm = 'WOEEncoder',
                    data_dependent = {'cols': 'categoricals'})},
            'mix': {
                'polynomial': AnalystTechnique(
                    name = 'polynomial_mixer',
                    module = 'sklearn.preprocessing',
                    algorithm = 'PolynomialFeatures',
                    default = {
                        'degree': 2,
                        'interaction_only': True,
                        'include_bias': True}),
                'quotient': AnalystTechnique(
                    name = 'quotient',
                    module = None,
                    algorithm = 'QuotientFeatures'),
                'sum': AnalystTechnique(
                    name = 'sum',
                    module = None,
                    algorithm = 'SumFeatures'),
                'difference': AnalystTechnique(
                    name = 'difference',
                    module = None,
                    algorithm = 'DifferenceFeatures')},
            'cleave': {
                'cleaver': AnalystTechnique(
                    name = 'cleaver',
                    module = 'simplify.analyst.algorithms',
                    algorithm = 'Cleaver')},
            'sample': {
                'adasyn': AnalystTechnique(
                    name = 'adasyn',
                    module = 'imblearn.over_sampling',
                    algorithm = 'ADASYN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'cluster': AnalystTechnique(
                    name = 'cluster',
                    module = 'imblearn.under_sampling',
                    algorithm = 'ClusterCentroids',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'knn': AnalystTechnique(
                    name = 'knn',
                    module = 'imblearn.under_sampling',
                    algorithm = 'AllKNN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'near_miss': AnalystTechnique(
                    name = 'near_miss',
                    module = 'imblearn.under_sampling',
                    algorithm = 'NearMiss',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'random_over': AnalystTechnique(
                    name = 'random_over',
                    module = 'imblearn.over_sampling',
                    algorithm = 'RandomOverSampler',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'random_under': AnalystTechnique(
                    name = 'random_under',
                    module = 'imblearn.under_sampling',
                    algorithm = 'RandomUnderSampler',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'smote': AnalystTechnique(
                    name = 'smote',
                    module = 'imblearn.over_sampling',
                    algorithm = 'SMOTE',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'smotenc': AnalystTechnique(
                    name = 'smotenc',
                    module = 'imblearn.over_sampling',
                    algorithm = 'SMOTENC',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    data_dependent = {
                        'categorical_features': 'categoricals_indices'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'smoteenn': AnalystTechnique(
                    name = 'smoteenn',
                    module = 'imblearn.combine',
                    algorithm = 'SMOTEENN',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample'),
                'smotetomek': AnalystTechnique(
                    name = 'smotetomek',
                    module = 'imblearn.combine',
                    algorithm = 'SMOTETomek',
                    default = {'sampling_strategy': 'auto'},
                    runtime = {'random_state': 'seed'},
                    fit_method = None,
                    transform_method = 'fit_resample')},
            'reduce': {
                'kbest': AnalystTechnique(
                    name = 'kbest',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectKBest',
                    default = {'k': 10, 'score_func': 'f_classif'},
                    selected = True),
                'fdr': AnalystTechnique(
                    name = 'fdr',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFdr',
                    default = {'alpha': 0.05, 'score_func': 'f_classif'},
                    selected = True),
                'fpr': AnalystTechnique(
                    name = 'fpr',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFpr',
                    default = {'alpha': 0.05, 'score_func': 'f_classif'},
                    selected = True),
                'custom': AnalystTechnique(
                    name = 'custom',
                    module = 'sklearn.feature_selection',
                    algorithm = 'SelectFromModel',
                    default = {'threshold': 'mean'},
                    runtime = {'estimator': 'algorithm'},
                    selected = True),
                'rank': AnalystTechnique(
                    name = 'rank',
                    module = 'simplify.critic.rank',
                    algorithm = 'RankSelect',
                    selected = True),
                'rfe': AnalystTechnique(
                    name = 'rfe',
                    module = 'sklearn.feature_selection',
                    algorithm = 'RFE',
                    default = {'n_features_to_select': 10, 'step': 1},
                    runtime = {'estimator': 'algorithm'},
                    selected = True),
                'rfecv': AnalystTechnique(
                    name = 'rfecv',
                    module = 'sklearn.feature_selection',
                    algorithm = 'RFECV',
                    default = {'n_features_to_select': 10, 'step': 1},
                    runtime = {'estimator': 'algorithm'},
                    selected = True)}}
        model_options = {
            'classify': {
                'adaboost': AnalystTechnique(
                    name = 'adaboost',
                    module = 'sklearn.ensemble',
                    algorithm = 'AdaBoostClassifier',
                    transform_method = None),
                'baseline_classifier': AnalystTechnique(
                    name = 'baseline_classifier',
                    module = 'sklearn.dummy',
                    algorithm = 'DummyClassifier',
                    required = {'strategy': 'most_frequent'},
                    transform_method = None),
                'logit': AnalystTechnique(
                    name = 'logit',
                    module = 'sklearn.linear_model',
                    algorithm = 'LogisticRegression',
                    transform_method = None),
                'random_forest': AnalystTechnique(
                    name = 'random_forest',
                    module = 'sklearn.ensemble',
                    algorithm = 'RandomForestClassifier',
                    transform_method = None),
                'svm_linear': AnalystTechnique(
                    name = 'svm_linear',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'linear', 'probability': True},
                    transform_method = None),
                'svm_poly': AnalystTechnique(
                    name = 'svm_poly',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'poly', 'probability': True},
                    transform_method = None),
                'svm_rbf': AnalystTechnique(
                    name = 'svm_rbf',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'rbf', 'probability': True},
                    transform_method = None),
                'svm_sigmoid': AnalystTechnique(
                    name = 'svm_sigmoid ',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'sigmoid', 'probability': True},
                    transform_method = None),
                'tensorflow': AnalystTechnique(
                    name = 'tensorflow',
                    module = 'tensorflow',
                    algorithm = None,
                    default = {
                        'batch_size': 10,
                        'epochs': 2},
                    transform_method = None),
                'xgboost': AnalystTechnique(
                    name = 'xgboost',
                    module = 'xgboost',
                    algorithm = 'XGBClassifier',
                    data_dependent = 'scale_pos_weight',
                    transform_method = None)},
            'cluster': {
                'affinity': AnalystTechnique(
                    name = 'affinity',
                    module = 'sklearn.cluster',
                    algorithm = 'AffinityPropagation',
                    transform_method = None),
                'agglomerative': AnalystTechnique(
                    name = 'agglomerative',
                    module = 'sklearn.cluster',
                    algorithm = 'AgglomerativeClustering',
                    transform_method = None),
                'birch': AnalystTechnique(
                    name = 'birch',
                    module = 'sklearn.cluster',
                    algorithm = 'Birch',
                    transform_method = None),
                'dbscan': AnalystTechnique(
                    name = 'dbscan',
                    module = 'sklearn.cluster',
                    algorithm = 'DBSCAN',
                    transform_method = None),
                'kmeans': AnalystTechnique(
                    name = 'kmeans',
                    module = 'sklearn.cluster',
                    algorithm = 'KMeans',
                    transform_method = None),
                'mean_shift': AnalystTechnique(
                    name = 'mean_shift',
                    module = 'sklearn.cluster',
                    algorithm = 'MeanShift',
                    transform_method = None),
                'spectral': AnalystTechnique(
                    name = 'spectral',
                    module = 'sklearn.cluster',
                    algorithm = 'SpectralClustering',
                    transform_method = None),
                'svm_linear': AnalystTechnique(
                    name = 'svm_linear',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM',
                    transform_method = None),
                'svm_poly': AnalystTechnique(
                    name = 'svm_poly',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM',
                    transform_method = None),
                'svm_rbf': AnalystTechnique(
                    name = 'svm_rbf',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM,',
                    transform_method = None),
                'svm_sigmoid': AnalystTechnique(
                    name = 'svm_sigmoid',
                    module = 'sklearn.cluster',
                    algorithm = 'OneClassSVM',
                    transform_method = None)},
            'regress': {
                'adaboost': AnalystTechnique(
                    name = 'adaboost',
                    module = 'sklearn.ensemble',
                    algorithm = 'AdaBoostRegressor',
                    transform_method = None),
                'baseline_regressor': AnalystTechnique(
                    name = 'baseline_regressor',
                    module = 'sklearn.dummy',
                    algorithm = 'DummyRegressor',
                    required = {'strategy': 'mean'},
                    transform_method = None),
                'bayes_ridge': AnalystTechnique(
                    name = 'bayes_ridge',
                    module = 'sklearn.linear_model',
                    algorithm = 'BayesianRidge',
                    transform_method = None),
                'lasso': AnalystTechnique(
                    name = 'lasso',
                    module = 'sklearn.linear_model',
                    algorithm = 'Lasso',
                    transform_method = None),
                'lasso_lars': AnalystTechnique(
                    name = 'lasso_lars',
                    module = 'sklearn.linear_model',
                    algorithm = 'LassoLars',
                    transform_method = None),
                'ols': AnalystTechnique(
                    name = 'ols',
                    module = 'sklearn.linear_model',
                    algorithm = 'LinearRegression',
                    transform_method = None),
                'random_forest': AnalystTechnique(
                    name = 'random_forest',
                    module = 'sklearn.ensemble',
                    algorithm = 'RandomForestRegressor',
                    transform_method = None),
                'ridge': AnalystTechnique(
                    name = 'ridge',
                    module = 'sklearn.linear_model',
                    algorithm = 'Ridge',
                    transform_method = None),
                'svm_linear': AnalystTechnique(
                    name = 'svm_linear',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'linear', 'probability': True},
                    transform_method = None),
                'svm_poly': AnalystTechnique(
                    name = 'svm_poly',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'poly', 'probability': True},
                    transform_method = None),
                'svm_rbf': AnalystTechnique(
                    name = 'svm_rbf',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'rbf', 'probability': True},
                    transform_method = None),
                'svm_sigmoid': AnalystTechnique(
                    name = 'svm_sigmoid ',
                    module = 'sklearn.svm',
                    algorithm = 'SVC',
                    required = {'kernel': 'sigmoid', 'probability': True},
                    transform_method = None),
                'xgboost': AnalystTechnique(
                    name = 'xgboost',
                    module = 'xgboost',
                    algorithm = 'XGBRegressor',
                    data_dependent = 'scale_pos_weight',
                    transform_method = None)}}
        gpu_options = {
            'classify': {
                'forest_inference': AnalystTechnique(
                    name = 'forest_inference',
                    module = 'cuml',
                    algorithm = 'ForestInference',
                    transform_method = None),
                'random_forest': AnalystTechnique(
                    name = 'random_forest',
                    module = 'cuml',
                    algorithm = 'RandomForestClassifier',
                    transform_method = None),
                'logit': AnalystTechnique(
                    name = 'logit',
                    module = 'cuml',
                    algorithm = 'LogisticRegression',
                    transform_method = None)},
            'cluster': {
                'dbscan': AnalystTechnique(
                    name = 'dbscan',
                    module = 'cuml',
                    algorithm = 'DBScan',
                    transform_method = None),
                'kmeans': AnalystTechnique(
                    name = 'kmeans',
                    module = 'cuml',
                    algorithm = 'KMeans',
                    transform_method = None)},
            'regressor': {
                'lasso': AnalystTechnique(
                    name = 'lasso',
                    module = 'cuml',
                    algorithm = 'Lasso',
                    transform_method = None),
                'ols': AnalystTechnique(
                    name = 'ols',
                    module = 'cuml',
                    algorithm = 'LinearRegression',
                    transform_method = None),
                'ridge': AnalystTechnique(
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
