
from dataclasses import dataclass

from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering,
                             Birch, DBSCAN, KMeans, MeanShift,
                             SpectralClustering)
from sklearn.svm import OneClassSVM

from simplify.core.base import SimpleTechnique


@dataclass
class Clusterer(SimpleTechnique):
    """Applies machine learning algorithms based upon user selections."""


    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'clusterer'

    def __post_init__(self):

        super().__post_init__()
        return self

    def draft(self):
        self.options = {'affinity' : AffinityPropagation,
                        'agglomerative' : AgglomerativeClustering,
                        'birch' : Birch,
                        'dbscan' : DBSCAN,
                        'kmeans' : KMeans,
                        'mean_shift' : MeanShift,
                        'spectral' : SpectralClustering,
                        'svm_linear' : OneClassSVM,
                        'svm_poly' : OneClassSVM,
                        'svm_rbf' : OneClassSVM,
                        'svm_sigmoid' : OneClassSVM}
        return self

    def produce(self, ingredients):
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return self.algorithm