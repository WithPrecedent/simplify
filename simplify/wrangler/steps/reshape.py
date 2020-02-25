"""
.. module:: reshape
:synopsis: data shaper
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""


from dataclasses import dataclass

import pandas as pd

from simplify.core.definitionsetter import WranglerTechnique


@dataclass
class Reshape(WranglerTechnique):
    """Reshapes a DataFrame to wide or long form.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance and
            other methods and classes in the siMpLify package..
        auto_draft (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'scaler'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        return self

    def _long(self, df):
        """A simple wrapper method for pandas wide_to_long method using more
        intuitive parameter names than 'i' and 'j'.
        """
        df = (pd.wide_to_long(df,
                              stubnames = self.stubs,
                              i = self.id_column,
                              j = self.values,
                              sep = self.separator).reset_index())
        return df

    def _wide(self, df):
        """A simple wrapper method for pandas pivot method named as
        corresponding method to reshape_long.
        """
        df = (df.pivot(index = self.id_column,
                       columns = self.stubs,
                       values = self.values).reset_index())
        return df


    def publish(self, dataset):
        dataset.df = getattr(self, '_' + self.shape_type)(dataset.df)
        return dataset