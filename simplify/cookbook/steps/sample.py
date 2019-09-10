
from dataclasses import dataclass

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import (AllKNN, ClusterCentroids, NearMiss,
                                     RandomUnderSampler)

from ..cookbook_step import CookbookStep


@dataclass
class Sample(CookbookStep):
    """Synthetically resamples data according to selected algorithm."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'sampler'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _define(self):
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

    def _add_parameters(self, ingredients, columns = None):
        if self.technique in ['smotenc']:
            if columns:
                cat_features = self._get_indices(ingredients.x, columns)
                self.parameters.update({'categorical_features' : cat_features})
            else:
                cat_features = self._get_indices(ingredients.x,
                                                 ingredients.categoricals)
        return self

    def fit(self, x, y, columns = None):
        self._add_parameters(x, columns)
        return self

    def fit_transform(self, x, y):
        self.fit(x, y)
        x = self.transform(x, y)
        return x

    def start(self, ingredients, recipe, columns = None):
        if self.technique != 'none':
            self._add_parameters(ingredients.x, columns)
            if recipe.data_to_use in ['full']:
                self._store_feature_names(ingredients.x, ingredients.y)
                resampled_x, resampled_y = self.algorithm.fit_resample(
                        ingredients.x, ingredients.y)
                ingredients.x, ingredients.y = self._get_feature_names(
                        resampled_x, resampled_y)
            else:
                self._store_feature_names(ingredients.x_train,
                                          ingredients.y_train)
                resampled_x, resampled_y = self.algorithm.fit_resample(
                        ingredients.x_train, ingredients.y_train)
                ingredients.x_train, ingredients.y_train = (
                        self._get_feature_names(resampled_x, resampled_y))
        return ingredients

    def transform(self, x, y):
        x = self.algorithm.fit_resample(x, y)
        return x