"""
.. module:: scaler
:synopsis: scales or bins numerical features
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.typesetter import SimpleDirector
from simplify.core.typesetter import Option


@dataclass
class Scaler(SimpleDirector):
    """Scales numerical data in a DataFrame stored in Ingredients instance.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    name: str = 'scaler'

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self._options = SimpleCatalog(options = {
            'bins': Option(
                name = 'bins',
                module = 'sklearn.preprocessing',
                algorithm = 'KBinsDiscretizer',
                default = {
                    'encode': 'ordinal',
                    'strategy': 'uniform',
                    'n_bins': 5},
                selected = True),
            # 'gauss': Option(
            #     name = 'gauss',
            #     module = None,
            #     algorithm = Gaussify,
            #     default = {'standardize': False, 'copy': False},
            #     required = {'rescaler': self.standard},
            #     selected = True),
            'maxabs': Option(
                name = 'maxabs',
                module = 'sklearn.preprocessing',
                algorithm = 'MaxAbsScaler',
                default = {'copy': False},
                selected = True),
            'minmax': Option(
                name = 'minmax',
                module = 'sklearn.preprocessing',
                algorithm = 'MinMaxScaler',
                default = {'copy': False},
                selected = True),
            'normalize': Option(
                name = 'normalize',
                module = 'sklearn.preprocessing',
                algorithm = 'Normalizer',
                default = {'copy': False},
                selected = True),
            'quantile': Option(
                name = 'quantile',
                module = 'sklearn.preprocessing',
                algorithm = 'QuantileTransformer',
                default = {'copy': False},
                selected = True),
            'robust': Option(
                name = 'robust',
                module = 'sklearn.preprocessing',
                algorithm = 'RobustScaler',
                default = {'copy': False},
                selected = True),
            'standard': Option(
                name = 'standard',
                module = 'sklearn.preprocessing',
                algorithm = 'StandardScaler',
                default = {'copy': False},
                selected = True)}
        return self


# @dataclass
# class Gaussify(Algorithm):
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
#         self.idea_sections = ['chef']
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
