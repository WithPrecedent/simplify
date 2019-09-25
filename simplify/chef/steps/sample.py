
from dataclasses import dataclass

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import (AllKNN, ClusterCentroids, NearMiss,
                                     RandomUnderSampler)

from simplify.core.base import SimpleStep
from simplify.core.decorators import numpy_shield


@dataclass
class Sample(SimpleStep):
    """Synthetically resamples data according to selected algorithm.

    Args:
        technique(str): name of technique - it should always be 'gauss'
        parameters(dict): dictionary of parameters to pass to selected technique
            algorithm.
        auto_finalize(bool): whether 'finalize' method should be called when the
            class is instanced. This should generally be set to True.
        name(str): name of class for matching settings in the Idea instance and
            for labeling the columns in files exported by Critic.
    """
    techniques : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'sampler'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options  = {'adasyn' : ADASYN,
                         'cluster' : ClusterCentroids,
                         'knn' : AllKNN,
                         'near_miss' : NearMiss,
                         'random_over' : RandomOverSampler,
                         'random_under' : RandomUnderSampler,
                         'smote' : SMOTE,
                         'smotenc' : SMOTENC,
                         'smoteenn' :  SMOTEENN,
                         'smotetomek' : SMOTETomek}
        self.default_parameters = {'sampling_strategy' : 'auto'}
        self.runtime_parameters = {'random_state' : self.seed}
        return self

    def _recheck_parameters(self, ingredients, columns = None):
        if self.technique in ['smotenc']:
            if columns:
                cat_features = self._get_indices(ingredients.x, columns)
                self.parameters.update({'categorical_features' : cat_features})
            else:
                cat_features = self._get_indices(ingredients.x,
                                                 ingredients.categoricals)
        return self

    def fit(self, x, y, columns = None):
        self._recheck_parameters(x, columns)
        return self

    def fit_transform(self, x, y):
        x = self.transform(x, y)
        return x

    @numpy_shield
    def produce(self, ingredients, plan = None, columns = None):
        self._recheck_parameters(ingredients.x, columns)
        if plan.data_to_use in ['full']:
            resampled_x, resampled_y = self.algorithm.fit_resample(
                    ingredients.x, ingredients.y)
            ingredients.x, ingredients.y = self._get_column_names(
                    resampled_x, resampled_y)
        else:
            resampled_x, resampled_y = self.algorithm.fit_resample(
                    ingredients.x_train, ingredients.y_train)
        return ingredients

    def transform(self, x, y):
        x = self.algorithm.fit_resample(x, y)
        return x