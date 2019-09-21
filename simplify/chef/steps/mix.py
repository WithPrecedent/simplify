
from dataclasses import dataclass

from category_encoders import PolynomialEncoder

from simplify.core.base import SimpleStep
from simplify.core.decorators import oven_mits

@dataclass
class Mix(SimpleStep):
    """Computes new features using different algorithms selected.

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
    name: str = 'mixer'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {'polynomial' : PolynomialEncoder,
                        'quotient' : self.quotient_features,
                        'sum' : self.sum_features,
                        'difference' : self.difference_features}
        self.default_parameters = {}
        return self

    def quotient_features(self):
        pass
        return self

    def sum_features(self):
        pass
        return self

    def difference_features(self):
        pass
        return self

    def finalize(self):
        pass
        return self

    @oven_mits
    def produce(self, ingredients, plan = None, columns = None):
        if not columns:
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