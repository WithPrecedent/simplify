"""
.. module:: analyst algorithms
:synopsis: custom algorithms for the analyst subpackage
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def auto_categorize(
        ingredient: 'Ingredient',
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[int] = 10) -> 'Ingredient':
    """Converts appropriate columns to 'categorical' type.

    The function automatically assesses each column to determine if it has less
    than 'threshold' unique values and is not boolean. If so, that column is
    converted to 'categorical' type.

    Args:
        ingredient ('Ingredient'): instance storing a pandas DataFrame.
        columns (Optional[Union[List[str], str]]): column names to be checked.
            Defaults to None. If not passed, all columns are checked.
        threshold (Optional[int]): number of unique values under which the
            column will be converted to 'categorical'. Defaults to 10.

    Raises:
        KeyError: if a column in 'columns' is not in 'ingredient'.

    """
    if not columns:
        columns = list(ingredient.datatypes.keys())
    for column in columns:
        try:
            if not column in ingredient.booleans:
                if ingredient[column].nunique() < threshold:
                    ingredient[column] = ingredient[column].astype('category')
                    ingredient.datatypes[column] = 'categorical'
        except KeyError:
            raise KeyError(' '.join([column, 'is not in ingredient']))
    return ingredient

def combine_rare(
        ingredient: 'Ingredient',
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0) -> 'Ingredient':
    """Converts rare categories to a single category.

    The threshold is defined as the percentage of total rows.

    Args:
        ingredient ('Ingredient'): instance storing a pandas DataFrame.
        columns (Optional[Union[List[str], str]]): column names to be checked.
            Defaults to None. If not passed, all 'categorical'columns are
            checked.
        threshold (Optional[float]): indicates the percentage of values in rows
            below which the categories are collapsed into a single category.
            Defaults to 0, meaning no categories are eliminated.

    Raises:
        KeyError: if a column in 'columns' is not in 'ingredient'.

    """
    if not columns:
        columns = singredient.categoricals
    for column in columns:
        try:
            counts = ingredient[column].value_counts()
            frequencies = (counts/counts.sum() * 100).lt(1)
            rare = frequencies[frequencies <= threshold].index
            ingredient[column].replace(rare , 'rare', inplace = True)
        except KeyError:
            raise KeyError(' '.join([column, 'is not in ingredient']))
    return ingredient

def decorrelate(
        ingredient: 'Ingredient',
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0.95) -> 'Ingredient':
    """Drops all but one column from highly correlated groups of columns.

    The threshold is based upon the .corr() method in pandas. 'columns' can
    include any datatype accepted by .corr(). If 'columns' is None, all
    columns in the DataFrame are tested.

    Args:
        ingredient ('Ingredient'): instance storing a pandas DataFrame.
        columns (Optional[Union[List[str], str]]): column names to be checked.
            Defaults to None. If not passed, all columns are checked.
        threshold (Optional[float]): the level of correlation using pandas corr
            method above which a column is dropped. The default threshold is
            0.95, consistent with a common p-value threshold used in social
            science research.

    """
    if not columns:
        columns = list(ingredient.datatypes.keys())
    try:
        corr_matrix = ingredient[columns].corr().abs()
    except TypeError:
        corr_matrix = ingredient.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    corrs = [col for col in upper.corrs if any(upper[col] > threshold)]
    ingredient.drop_columns(columns = corrs)
    return ingredient

def drop_infrequently_true(
        ingredient: 'Ingredient',
        columns: Optional[Union[List[str], str]] = None,
        threshold: Optional[float] = 0) -> 'Ingredient':
    """Drops boolean columns that rarely are True.

    This differs from the sklearn VarianceThreshold class because it is only
    concerned with rare instances of True and not False. This enables
    users to set a different variance threshold for rarely appearing
    information. 'threshold' is defined as the percentage of total rows (and
    not the typical variance formulas used in sklearn).

    Args:
        ingredient ('Ingredient'): instance storing a pandas DataFrame.
        columns (list or str): columns to check.
        threshold (float): the percentage of True values in a boolean column
            that must exist for the column to be kept.
    """
    if columns is None:
        columns = ingredient.booleans
    infrequents = []
    for column in listify(columns):
        try:
            if ingredient[column].mean() < threshold:
                infrequents.append(column)
        except KeyError:
            raise KeyError(' '.join([column, 'is not in ingredient']))
    ingredient.drop_columns(columns = infrequents)
    return ingredient

def smart_fill(
        ingredient: 'Ingredient',
        columns: Optional[Union[List[str], str]] = None) -> 'Ingredient':
    """Fills na values in a DataFrame with defaults based upon the datatype
    listed in 'all_datatypes'.

    Args:
        ingredient ('Ingredient'): instance storing a pandas DataFrame.
        columns (list): list of columns to fill missing values in. If no
            columns are passed, all columns are filled.

    Raises:
        KeyError: if column in 'columns' is not in 'ingredient'.

    """
    for column in self._check_columns(columns):
        try:
            default_value = self.all_datatypes.default_values[
                    self.columns[column]]
            ingredient[column].fillna(default_value, inplace = True)
        except KeyError:
            raise KeyError(' '.join([column, 'is not in ingredient']))
    return ingredient

def split_xy(
        ingredients: 'Ingredients',
        label: Optional[str] = 'label') -> 'Ingredients':
    """Splits ingredient into 'x' and 'y' based upon the label ('y' column) passed.

    Args:
        ingredient ('Ingredient'): instance storing a pandas DataFrame.
        label (str or list): name of column(s) to be stored in 'y'.'

    """
    ingredients.x = ingredient[list(ingredient.columns.values).remove(label)]
    ingredients.y = ingredient[label],
    ingredients.label_datatype = self.columns[label]
    ingredients._crosscheck_columns()
    singredients.state.change('train_test')
    return ingredients


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
#         self.positive_tool = self.tasks['box_cox'](
#                 method = 'box_cox', **self.parameters)
#         self.negative_tool = self.tasks['yeo_johnson'](
#                 method = 'yeo_johnson', **self.parameters)
#         return self

#     def publish(self, ingredients, columns = None):
#         if not columns:
#             columns = ingredients.numerics
#         for column in columns:
#             if ingredients.x[column].min() >= 0:
#                 ingredients.x[column] = self.positive_tool.fit_transform(
#                         ingredients.x[column])
#             else:
#                 ingredients.x[column] = self.negative_tool.fit_transform(
#                         ingredients.x[column])
#             ingredients.x[column] = self.rescaler.fit_transform(
#                     ingredients.x[column])
#         return ingredients

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



# def make_tensorflow_model(step: 'Technique', parameters: dict) -> None:
#     algorithm = None
#     return algorithm


#    def _downcast_features(self, ingredients):
#        dataframes = ['x_train', 'x_test']
#        number_types = ['uint', 'int', 'float']
#        feature_bits = ['64', '32', '16']
#        for ingredient in dataframes:
#            for column in ingredient.columns.keys():
#                if (column in ingredients.floats
#                        or column in ingredients.integers):
#                    for number_type in number_types:
#                        for feature_bit in feature_bits:
#                            try:
#                                ingredient[column] = ingredient[column].astype()

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

