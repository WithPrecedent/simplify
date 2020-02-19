"""
.. module:: analyst algorithms
:synopsis: custom algorithms for the analyst subpackage
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd


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
    for column in listify(columns):
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



# @dataclass
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

# @dataclass
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

# @dataclass
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

