"""
.. module:: cluster
:synopsis: unsupervised (clustering) machine learning algorithms
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

from simplify.core.technique import ChefTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'affinity': ['sklearn.cluster', 'AffinityPropagation'],
    'agglomerative': ['sklearn.cluster', 'AgglomerativeClustering'],
    'birch': ['sklearn.cluster', 'Birch'],
    'dbscan': ['sklearn.cluster', 'DBSCAN'],
    'kmeans': ['sklearn.cluster', 'KMeans'],
    'mean_shift': ['sklearn.cluster', 'MeanShift'],
    'spectral': ['sklearn.cluster', 'SpectralClustering'],
    'svm_linear': ['sklearn.cluster', 'OneClassSVM'],
    'svm_poly': ['sklearn.cluster', 'OneClassSVM'],
    'svm_rbf': ['sklearn.cluster', 'OneClassSVM,'],
    'svm_sigmoid': ['sklearn.cluster', 'OneClassSVM']}


@dataclass
class Cluster(ChefTechnique):
    """Applies machine learning algorithms based upon user selections.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_draft (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    auto_draft: bool = True
    name: str = 'clusterer'
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        self.idea_sections = ['chef']
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
        self._get_conditional_options()
        return self

    def implement(self, ingredients):
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        return self.algorithm