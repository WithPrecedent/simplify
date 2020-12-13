"""
analyst: modeling and analytic classes and functions
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    

    
"""

from __future__ import annotations
import copy
import dataclasses
import functools
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import numpy as np
import pandas as pd
import scipy
import sklearn

import simplify
import sourdough



def auto_categorize(
        data: 'Data',
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[int] = 10) -> 'Data':
    """Converts appropriate columns to 'categorical' type.

    The function automatically assesses each column to determine if it has less
    than 'threshold' unique values and is not boolean. If so, that column is
    converted to 'categorical' type.

    Args:
        data ('Data'): instance storing a pandas DataFrame.
        columns (Optional[Union[List[str], str]]): column names to be checked.
            Defaults to None. If not passed, all columns are checked.
        threshold (Optional[int]): number of unique values under which the
            column will be converted to 'categorical'. Defaults to 10.

    Raises:
        KeyError: if a column in 'columns' is not in 'data'.

    """
    if not columns:
        columns = list(data.datatypes.keys())
    for column in columns:
        try:
            if not column in data.booleans:
                if data[column].nunique() < threshold:
                    data[column] = data[column].astype('category')
                    data.datatypes[column] = 'categorical'
        except KeyError:
            raise KeyError(' '.join([column, 'is not in data']))
    return data

def combine_rare(
        data: 'Data',
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0) -> 'Data':
    """Converts rare categories to a single category.

    The threshold is defined as the percentage of total rows.

    Args:
        data ('Data'): instance storing a pandas DataFrame.
        columns (Optional[Union[List[str], str]]): column names to be checked.
            Defaults to None. If not passed, all 'categorical'columns are
            checked.
        threshold (Optional[float]): indicates the percentage of values in rows
            below which the categories are collapsed into a single category.
            Defaults to 0, meaning no categories are eliminated.

    Raises:
        KeyError: if a column in 'columns' is not in 'data'.

    """
    if not columns:
        columns = sdata.categoricals
    for column in columns:
        try:
            counts = data[column].value_counts()
            frequencies = (counts/counts.sum() * 100).lt(1)
            rare = frequencies[frequencies <= threshold].index
            data[column].replace(rare , 'rare', inplace = True)
        except KeyError:
            raise KeyError(' '.join([column, 'is not in data']))
    return data

def decorrelate(
        data: 'Data',
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0.95) -> 'Data':
    """Drops all but one column from highly correlated groups of columns.

    The threshold is based upon the .corr() method in pandas. 'columns' can
    include any datatype accepted by .corr(). If 'columns' is None, all
    columns in the DataFrame are tested.

    Args:
        data ('Data'): instance storing a pandas DataFrame.
        columns (Optional[Union[List[str], str]]): column names to be checked.
            Defaults to None. If not passed, all columns are checked.
        threshold (Optional[float]): the level of correlation using pandas corr
            method above which a column is dropped. The default threshold is
            0.95, consistent with a common p-value threshold used in social
            science research.

    """
    if not columns:
        columns = list(data.datatypes.keys())
    try:
        corr_matrix = data[columns].corr().abs()
    except TypeError:
        corr_matrix = data.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    corrs = [col for col in upper.corrs if any(upper[col] > threshold)]
    data.drop_columns(columns = corrs)
    return data

def drop_infrequently_true(
        data: 'Data',
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0) -> 'Data':
    """Drops boolean columns that rarely are True.

    This differs from the sklearn VarianceThreshold class because it is only
    concerned with rare instances of True and not False. This enables
    users to set a different variance threshold for rarely appearing
    information. 'threshold' is defined as the percentage of total rows (and
    not the typical variance formulas used in sklearn).

    Args:
        data ('Data'): instance storing a pandas DataFrame.
        columns (list or str): columns to check.
        threshold (float): the percentage of True values in a boolean column
            that must exist for the column to be kept.
    """
    if columns is None:
        columns = data.booleans
    infrequents = []
    for column in utilities.listify(columns):
        try:
            if data[column].mean() < threshold:
                infrequents.append(column)
        except KeyError:
            raise KeyError(' '.join([column, 'is not in data']))
    data.drop_columns(columns = infrequents)
    return data

def smart_fill(
        data: 'Data',
        columns: Optional[Union[List[str], str]] = None) -> 'Data':
    """Fills na values in a DataFrame with defaults based upon the datatype
    listed in 'all_datatypes'.

    Args:
        data ('Data'): instance storing a pandas DataFrame.
        columns (list): list of columns to fill missing values in. If no
            columns are passed, all columns are filled.

    Raises:
        KeyError: if column in 'columns' is not in 'data'.

    """
    for column in self._check_columns(columns):
        try:
            default_value = self.all_datatypes.default_values[
                    self.columns[column]]
            data[column].fillna(default_value, inplace = True)
        except KeyError:
            raise KeyError(' '.join([column, 'is not in data']))
    return data



# @dataclasses.dataclass
# class Gaussify(TechniqueOutline):
#     """Transforms data columns to more gaussian distribution.

#     The particular method applied is chosen between 'box-cox' and 'yeo-johnson'
#     based on whether the particular data column has values below zero.

#     Args:
#         step(str): name of step used.
#         parameters(dict): dictionary of parameters to pass to selected
#             algorithm.
#         name(str): name of class for matching settings in the Idea instance
#             and for labeling the columns in files exported by Critic.
#         auto_draft(bool): whether 'finalize' method should be called when
#             the class is instanced. This should generally be set to True.
#     """

#     step: str = 'box-cox and yeo-johnson'
#     parameters: object = None
#     name: str = 'gaussifier'

#     def __post_init__(self) -> None:
#         self.idea_sections = ['analyst']
#         super().__post_init__()
#         return self

#     def draft(self) -> None:
#         self.rescaler = self.parameters['rescaler'](
#                 copy = self.parameters['copy'])
#         del self.parameters['rescaler']
#         self._publish_parameters()
#         self.positive_tool = self.workers['box_cox'](
#                 method = 'box_cox', **self.parameters)
#         self.negative_tool = self.workers['yeo_johnson'](
#                 method = 'yeo_johnson', **self.parameters)
#         return self

#     def publish(self, dataset, columns = None):
#         if not columns:
#             columns = dataset.numerics
#         for column in columns:
#             if dataset.x[column].min() >= 0:
#                 dataset.x[column] = self.positive_tool.fit_transform(
#                         dataset.x[column])
#             else:
#                 dataset.x[column] = self.negative_tool.fit_transform(
#                         dataset.x[column])
#             dataset.x[column] = self.rescaler.fit_transform(
#                     dataset.x[column])
#         return dataset

# @dataclasses.dataclass
# class CompareCleaves(TechniqueOutline):
#     """[summary]

#     Args:
#         step (str):
#         parameters (dict):
#         space (dict):
#     """
#     step: str
#     parameters: object
#     space: object

#     def __post_init__(self) -> None:
#         self.idea_sections = ['analyst']
#         super().__post_init__()
#         return self

    # def _cleave(self, dataset):
    #     if self.step != 'all':
    #         cleave = self.workers[self.step]
    #         drop_list = [i for i in self.test_columns if i not in cleave]
    #         for col in drop_list:
    #             if col in dataset.x_train.columns:
    #                 dataset.x_train.drop(col, axis = 'columns',
    #                                          inplace = True)
    #                 dataset.x_test.drop(col, axis = 'columns',
    #                                         inplace = True)
    #     return dataset

    # def _publish_cleaves(self):
    #     for group, columns in self.workers.items():
    #         self.test_columns.extend(columns)
    #     if self.parameters['include_all']:
    #         self.workers.update({'all': self.test_columns})
    #     return self

    # def add(self, cleave_group, columns):
    #     """For the cleavers in siMpLify, this step alows users to manually
    #     add a new cleave group to the cleaver dictionary.
    #     """
    #     self.workers.update({cleave_group: columns})
    #     return self


#        self.scorers = {'f_classif': f_classif,
#                        'chi2': chi2,
#                        'mutual_class': mutual_info_classif,
#                        'mutual_regress': mutual_info_regression}

# @dataclasses.dataclass
# class CombineCleaves(TechniqueOutline):
#     """[summary]

#     Args:
#         step (str):
#         parameters (dict):
#         space (dict):
#     """
#     step: str
#     parameters: object
#     space: object

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self


# @dataclasses.dataclass
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


# @dataclasses.dataclass
# class SearchAnalystTechnique(AnalystTechnique):
#     """[summary]

#     Args:
#         object ([type]): [description]
#     """
#     step: str
#     algorithm: object
#     parameters: object
#     data_dependent: object = None
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


    # # @numpy_shield
    # def publish_reduce(self, dataset, plan = None, estimator = None):
    #     if not estimator:
    #         estimator = plan.model.algorithm
    #     self._set_parameters(estimator)
    #     self.algorithm = self.workers[self.step](**self.parameters)
    #     if len(dataset.x_train.columns) > self.num_features:
    #         self.algorithm.fit(dataset.x_train, dataset.y_train)
    #         mask = ~self.algorithm.get_support()
    #         dataset.drop_columns(df = dataset.x_train, mask = mask)
    #         dataset.drop_columns(df = dataset.x_test, mask = mask)
    #     return dataset


    # def _set_reduce_parameters(self, estimator):
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


# def make_tensorflow_model(step: 'Technique', parameters: dict) -> None:
#     algorithm = None
#     return algorithm


#    def _downcast_features(self, dataset):
#        dataframes = ['x_train', 'x_test']
#        number_types = ['uint', 'int', 'float']
#        feature_bits = ['64', '32', '16']
#        for data in dataframes:
#            for column in data.columns.keys():
#                if (column in dataset.floats
#                        or column in dataset.integers):
#                    for number_type in number_types:
#                        for feature_bit in feature_bits:
#                            try:
#                                data[column] = data[column].astype()

#
#    def _set_feature_types(self):
#        self.type_interface = {'boolean': tensorflow.bool,
#                               'float': tensorflow.float16,
#                               'integer': tensorflow.int8,
#                               'string': object,
#                               'categorical': CategoricalDtype,
#                               'list': list,
#                               'datetime': datetime64,
#                               'timedelta': timedelta}


#    def _tensor_flow_model(self):
#        from keras.models import Sequential
#        from keras.layers import Dense, Dropout, Activation, Flatten
#        classifier = Sequential()
#        classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
#            activation = 'relu', input_dim = 30))
#        classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
#            activation = 'relu'))
#        classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
#            activation = 'sigmoid'))
#        classifier.compile(optimizer = 'adam',
#                           loss = 'binary_crossentropy',
#                           metrics = ['accuracy'])
#        return classifier
#        model = Sequential()
#        model.add(Activation('relu'))
#        model.add(Activation('relu'))
#        model.add(Dropout(0.25))
#        model.add(Flatten())
#        for layer_size in self.parameters['dense_layer_sizes']:
#            model.add(Dense(layer_size))
#            model.add(Activation('relu'))
#        model.add(Dropout(0.5))
#        model.add(Dense(2))
#        model.add(Activation('softmax'))
#        model.compile(loss = 'categorical_crossentropy',
#                      optimizer = 'adadelta',
#                      metrics = ['accuracy'])
#        return model



# def make_torch_model(step: 'Technique', parameters: dict) -> None:
#     algorithm = None
#     return algorithm


# def make_stan_model(step: 'Technique', parameters: dict) -> None:
#     algorithm = None
#     return algorithm




""" Book Subclasses """

@dataclasses.dataclass
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
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'cookbook')
    chapters: Optional[List['Chapter']] = dataclasses.field(default_factory = list)
    iterable: Optional[str] = dataclasses.field(default_factory = lambda: 'recipes')


@dataclasses.dataclass
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
    steps: Optional[List[Tuple[str, str]]] = dataclasses.field(default_factory = list)
    techniques: Optional[List['Technique']] = dataclasses.field(default_factory = list)

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
        if isinstance(utilities.listify(techniques)[0], Tuple):
            self.steps.extend(utilities.listify(techniques))
        else:
            self.techniques.extend(utilities.listify(techniques))
        return self



@dataclasses.dataclass
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
        algorithm (Optional[object]): process object which executes the primary
            method of a class instance. Defaults to None.
        parameters (Optional[Dict[str, Any]]): parameters to be attached to
            'algorithm' when 'algorithm' is instanced. Defaults to an empty
            dictionary.

    """
    name: Optional[str] = None
    step: Optional[str] = None
    module: Optional[str] = None
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = dataclasses.field(default_factory = dict)
    default: Optional[Dict[str, Any]] = dataclasses.field(default_factory = dict)
    required: Optional[Dict[str, Any]] = dataclasses.field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = dataclasses.field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    data_dependent: Optional[Dict[str, str]] = dataclasses.field(default_factory = dict)
    parameter_space: Optional[Dict[str, List[Union[int, float]]]] = dataclasses.field(
        default_factory = dict)
    fit_method: Optional[str] = dataclasses.field(default_factory = lambda: 'fit')
    transform_method: Optional[str] = dataclasses.field(
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

@dataclasses.dataclass
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

@dataclasses.dataclass
class AnalystScholar(Scholar):
    """Applies a 'Cookbook' instance to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        print('test idea in analyst', self.idea)
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


@dataclasses.dataclass
class AnalystFinisher(Finisher):
    """Finalizes 'Technique' instances with data-dependent parameters.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

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


@dataclasses.dataclass
class AnalystSpecialist(Specialist):
    """Base class for applying 'Technique' instances to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

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

@dataclasses.dataclass
class Tools(SimpleRepository):
    """A dictonary of Tool options for the Analyst subpackage.

    Args:
        idea (Optional[Idea]): shared 'Idea' instance with project settings.

    """
    idea: Optional[core.Idea] = None

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

        
""" Decorators """

def numpy_shield(process: Callable) -> Callable:
    """
    """
    @functools.wraps(process)
    def wrapper(*args, **kwargs):
        call_signature = inspect.signature(process)
        arguments = dict(call_signature.bind(*args, **kwargs).arguments)
        try:
            x_columns = list(arguments['x'].columns.values)
            result = process(*args, **kwargs)
            if isinstance(result, np.ndarray):
                result = pd.DataFrame(result, columns = x_columns)
        except KeyError:
            result = process(*args, **kwargs)
        return result
    return wrapper


@dataclasses.dataclass
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
        import_folder (Optional[str]): name of attribute in 'clerk' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'clerk' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.

    """
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'analyst')
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.analyst.analyst')
    book: Optional[str] = dataclasses.field(default_factory = lambda: 'Cookbook')
    chapter: Optional[str] = dataclasses.field(default_factory = lambda: 'Recipe')
    technique: Optional[str] = dataclasses.field(default_factory = lambda: 'Tool')
    publisher: Optional[str] = dataclasses.field(
        default_factory = lambda: 'AnalystPublisher')
    scholar: Optional[str] = dataclasses.field(default_factory = lambda: 'AnalystScholar')
    options: Optional[str] = dataclasses.field(default_factory = lambda: 'Tools')
    idea: Optional[core.Idea] = None
