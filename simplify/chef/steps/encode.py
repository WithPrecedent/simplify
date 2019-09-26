
from dataclasses import dataclass

from category_encoders import (BackwardDifferenceEncoder, BaseNEncoder,
                               BinaryEncoder, HashingEncoder, HelmertEncoder,
                               LeaveOneOutEncoder, OneHotEncoder,
                               OrdinalEncoder, SumEncoder, TargetEncoder)

from simplify.core.base import SimpleStep
from simplify.core.decorators import numpy_shield


@dataclass
class Encode(SimpleStep):
    """Encodes categorical variables according to a selected algorithm.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique : str = ''
    parameters : object = None
    name : str = 'encoder'
    auto_finalize : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.options = {'backward' : BackwardDifferenceEncoder,
                        'basen' : BaseNEncoder,
                        'binary' : BinaryEncoder,
                        'dummy' : OneHotEncoder,
                        'hashing' : HashingEncoder,
                        'helmert' : HelmertEncoder,
                        'loo' : LeaveOneOutEncoder,
                        'ordinal' : OrdinalEncoder,
                        'sum' : SumEncoder,
                        'target' : TargetEncoder}
        return self

    def finalize(self):
        pass
        return self

    @numpy_shield
    def produce(self, ingredients, plan = None, columns = None):
        if columns is None:
            columns = ingredients.encoders
        if columns:
            self.runtime_parameters.update({'cols' : columns})
        super().finalize()
        self.algorithm.fit(ingredients.x, ingredients.y)
        self.algorithm.transform(
                ingredients.x_train).reset_index(drop = True)
        self.algorithm.transform(
                ingredients.x_test).reset_index(drop = True)
        return ingredients