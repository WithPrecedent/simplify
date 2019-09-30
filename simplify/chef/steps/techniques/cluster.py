
from dataclasses import dataclass

from simplify.core.base import SimpleTechnique


@dataclass
class Cluster(SimpleTechnique):
    """Applies machine learning algorithms based upon user selections.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    auto_finalize: bool = True
    name: str = 'clusterer'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        self.options = {
                'affinity': ['sklearn.cluster', 'AffinityPropagation'],
                'agglomerative': ['sklearn.cluster',
                                   'AgglomerativeClustering'],
                'birch': ['sklearn.cluster', 'Birch'],
                'dbscan': ['sklearn.cluster', 'DBSCAN'],
                'kmeans': ['sklearn.cluster', 'KMeans'],
                'mean_shift': ['sklearn.cluster', 'MeanShift'],
                'spectral': ['sklearn.cluster', 'SpectralClustering'],
                'svm_linear': ['sklearn.cluster', 'OneClassSVM'],
                'svm_poly': ['sklearn.cluster', 'OneClassSVM'],
                'svm_rbf': ['sklearn.cluster', 'OneClassSVM,'],
                'svm_sigmoid': ['sklearn.cluster', 'OneClassSVM']}
        return self

    def produce(self, ingredients):
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return self.algorithm