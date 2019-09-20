
from dataclasses import dataclass

from category_encoders import (BackwardDifferenceEncoder, BaseNEncoder,
                               BinaryEncoder, HashingEncoder, HelmertEncoder,
                               LeaveOneOutEncoder, OneHotEncoder,
                               OrdinalEncoder, SumEncoder, TargetEncoder)

from simplify.core.base import SimpleStep


@dataclass
class Encode(SimpleStep):
    """Encodes categorical variables according to selected algorithms.
    
    Args:
        technique(str): name of technique - it should always be 'gauss'
        parameters(dict): dictionary of parameters to pass to selected technique
            algorithm.
        auto_finalize(bool): whether 'finalize' method should be called when the
            class is instanced. This should generally be set to True.
        store_names(bool): whether this class requires the feature names to be
            stored before the 'finalize' and 'produce' methods or called and
            then restored after both are utilized. This should be set to True
            when the class is using numpy methods.
        name(str): name of class for matching settings in the Idea instance and
            for labeling the columns in files exported by Critic.
    """
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    store_names : bool = False
    name : str = 'encoder'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
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
        self.default_parameters = {}
        return self
    
    def finalize(self):
        pass
        return self
    
    def produce(self, ingredients, plan = None, columns = None):
        if self.technique != 'none':
            if not columns:
                columns = ingredients.encoders
            if columns:
                self.runtime_parameters.update({'cols' : columns})
                super(SimpleStep).finalize()
            self.algorithm.fit(ingredients.x, ingredients.y)
            self.algorithm.transform(
                    ingredients.x_train).reset_index(drop = True)
            self.algorithm.transform(
                    ingredients.x_test).reset_index(drop = True)
        return ingredients