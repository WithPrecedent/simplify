"""
.. module:: cluster
:synopsis: unsupervised (clustering) machine learning algorithms
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleTechnique


@dataclass
class Cluster(SimpleTechnique):
    """Applies machine learning algorithms based upon user selections.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    auto_publish: bool = True
    name: str = 'clusterer'

    def __post_init__(self):
        super().__post_init__()
        return self
    
    """ Private Methods """
    
    def _get_conditional_options(self):
        if self.gpu:
            self.options.update({
                'dbscan': ['cuml', 'DBScan'],
                'kmeans': ['cuml', 'KMeans']})
        return self
 
    """ Core siMpLify Methods """
       
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
        self._get_conditional_options()
        return self

    def implement(self, ingredients):
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return self.algorithm