"""
.. module:: modeler
:synopsis: applies machine learning and statistical models to data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.chef.composer import ChefAlgorithm
from simplify.chef.composer import ChefComposer as Composer
from simplify.chef.composer import ChefTechnique as Technique

Algorithm = ModelAlgorithm


@dataclass
class Modeler(Composer):
    """Splits data into training, testing, and/or validation datasets.
    """

    name: str = 'modeler'

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Private Methods """

    def _add_gpu_techniques(self):
        getattr(self, ''.join('_add_gpu_techniques_', self.model_type))()
        return self

    def _add_gpu_techniques_classify(self):
        self.forest_inference = Technique(
            name = 'forest_inference',
            module = 'cuml',
            algorithm = 'ForestInference')
        self.random_forest = Technique(
            name = 'random_forest',
            module = 'cuml',
            algorithm = 'RandomForestClassifier')
        self.logit = Technique(
            name = 'logit',
            module = 'cuml',
            algorithm = 'LogisticRegression')
        return self

    def _add_gpu_techniques_cluster(self):
        self.dbscan = Technique(
            name = 'dbscan',
            module = 'cuml',
            algorithm = 'DBScan')
        self.kmeans = Technique(
            name = 'kmeans',
            module = 'cuml',
            algorithm = 'KMeans')
        return self

    def _add_gpu_techniques_regress(self):
        self.lasso = Technique(
            name = 'lasso',
            module = 'cuml',
            algorithm = 'Lasso')
        self.ols = Technique(
            name = 'ols',
            module = 'cuml',
            algorithm = 'LinearRegression')
        self.ridge = Technique(
            name = 'ridge',
            module = 'cuml',
            algorithm = 'RidgeRegression')
        return self

    def _draft_classify(self):
        self.adaboost = Technique(
            name = 'adaboost',
            module = 'sklearn.ensemble',
            algorithm = 'AdaBoostClassifier')
        self.baseline_classifier = Technique(
            name = 'baseline_classifier',
            module = 'sklearn.dummy',
            algorithm = 'DummyClassifier',
            extras = {'strategy': 'most_frequent'})
        self.logit = Technique(
            name = 'logit',
            module = 'sklearn.linear_model',
            algorithm = 'LogisticRegression')
        self.random_forest = Technique(
            name = 'random_forest',
            module = 'sklearn.ensemble',
            algorithm = 'RandomForestClassifier')
        self.svm_linear = Technique(
            name = 'svm_linear',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            extras = {'kernel': 'linear', 'probability': True})
        self.svm_poly = Technique(
            name = 'svm_poly',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            extras = {'kernel': 'poly', 'probability': True})
        self.svm_rbf = Technique(
            name = 'svm_rbf',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            extras = {'kernel': 'rbf', 'probability': True})
        self.svm_sigmoid = Technique(
            name = 'svm_sigmoid ',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            extras = {'kernel': 'sigmoid', 'probability': True})
        self.tensorflow = Technique(
            name = 'tensorflow',
            module = 'tensorflow',
            algorithm = None,
            defaults = {
                'batch_size': 10,
                'epochs': 2})
        self.xgboost = Technique(
            name = 'xgboost',
            module = 'xgboost',
            algorithm = 'XGBClassifier',
            data_dependents = 'scale_pos_weight')
        return self

    def _draft_cluster(self):
        self.affinity = Technique(
            name = 'affinity',
            module = 'sklearn.cluster',
            algorithm = 'AffinityPropagation')
        self.agglomerative = Technique(
            name = 'agglomerative',
            module = 'sklearn.cluster',
            algorithm = 'AgglomerativeClustering')
        self.birch = Technique(
            name = 'birch',
            module = 'sklearn.cluster',
            algorithm = 'Birch')
        self.dbscan = Technique(
            name = 'dbscan',
            module = 'sklearn.cluster',
            algorithm = 'DBSCAN')
        self.kmeans = Technique(
            name = 'kmeans',
            module = 'sklearn.cluster',
            algorithm = 'KMeans')
        self.mean_shift = Technique(
            name = 'mean_shift',
            module = 'sklearn.cluster',
            algorithm = 'MeanShift')
        self.spectral = Technique(
            name = 'spectral',
            module = 'sklearn.cluster',
            algorithm = 'SpectralClustering')
        self.svm_linear = Technique(
            name = 'svm_linear',
            module = 'sklearn.cluster',
            algorithm = 'OneClassSVM')
        self.svm_poly = Technique(
            name = 'svm_poly',
            module = 'sklearn.cluster',
            algorithm = 'OneClassSVM')
        self.svm_rbf = Technique(
            name = 'svm_rbf',
            module = 'sklearn.cluster',
            algorithm = 'OneClassSVM,')
        self.svm_sigmoid = Technique(
            name = 'svm_sigmoid',
            module = 'sklearn.cluster',
            algorithm = 'OneClassSVM')
        return self

    def _draft_regress(self):
        self.adaboost = Technique(
            name = 'adaboost',
            module = 'sklearn.ensemble',
            algorithm = 'AdaBoostRegressor')
        self.baseline_regressor = Technique(
            name = 'baseline_regressor',
            module = 'sklearn.dummy',
            algorithm = 'DummyRegressor',
            extras = {'strategy': 'mean'})
        self.bayes_ridge = Technique(
            name = 'bayes_ridge',
            module = 'sklearn.linear_model',
            algorithm = 'BayesianRidge')
        self.lasso = Technique(
            name = 'lasso',
            module = 'sklearn.linear_model',
            algorithm = 'Lasso')
        self.lasso_lars = Technique(
            name = 'lasso_lars',
            module = 'sklearn.linear_model',
            algorithm = 'LassoLars')
        self.ols = Technique(
            name = 'ols',
            module = 'sklearn.linear_model',
            algorithm = 'LinearRegression')
        self.random_forest = Technique(
            name = 'random_forest',
            module = 'sklearn.ensemble',
            algorithm = 'RandomForestRegressor')
        self.ridge = Technique(
            name = 'ridge',
            module = 'sklearn.linear_model',
            algorithm = 'Ridge')
        self.svm_linear = Technique(
            name = 'svm_linear',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            extras = {'kernel': 'linear', 'probability': True})
        self.svm_poly = Technique(
            name = 'svm_poly',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            extras = {'kernel': 'poly', 'probability': True})
        self.svm_rbf = Technique(
            name = 'svm_rbf',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            extras = {'kernel': 'rbf', 'probability': True})
        self.svm_sigmoid = Technique(
            name = 'svm_sigmoid ',
            module = 'sklearn.svm',
            algorithm = 'SVC',
            extras = {'kernel': 'sigmoid', 'probability': True})
        self.xgboost = Technique(
            name = 'xgboost',
            module = 'xgboost',
            algorithm = 'XGBRegressor',
            data_dependents = 'scale_pos_weight')
        return self

    def _get_conditionals(self, technique: SimpleTechnique, parameters: dict):
        if technique.name in ['xgboost'] and self.gpu:
            parameters.update({'tree_method': 'gpu_exact'})
        elif technique.name in ['tensorflow']:
            algorithm = create_tensorflow_model(
                technique = technique,
                parameters = parameters)
        return parameters

    """ Core siMpLify Methods """

    def draft(self):
        getattr(self, ''.join('_draft_', self.model_type))()
        super().draft()
        return self


@dataclass
class ModelAlgorithm(ChefAlgorithm):
    """[summary]

    Args:
        object ([type]): [description]
    """
    technique: str
    parameters: object
    space: object

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Scikit-Learn Compatibility Methods """

    def fit_transform(self, x, y = None):
        error = 'fit_transform is not implemented for machine learning models'
        raise NotImplementedError(error)

    def transform(self, x, y = None):
        error = 'transform is not implemented for machine learning models'
        raise NotImplementedError(error)


def create_tensorflow_model(technique: Technique, parameters: dict):
    algorithm = None
    return algorithm


#    def _downcast_features(self, ingredients):
#        dataframes = ['x_train', 'x_test']
#        number_types = ['uint', 'int', 'float']
#        feature_bits = ['64', '32', '16']
#        for df in dataframes:
#            for column in df.columns.keys():
#                if (column in ingredients.floats
#                        or column in ingredients.integers):
#                    for number_type in number_types:
#                        for feature_bit in feature_bits:
#                            try:
#                                df[column] = df[column].astype()

#
#    def _set_feature_types(self):
#        self.type_interface = {'boolean': tensorflow.bool,
#                               'float': tensorflow.float16,
#                               'integer': tensorflow.int8,
#                               'string': object,
#                               'categorical': CategoricalDtype,
#                               'list': list,
#                               'datetime': datetime64,
#                               'timedelta': timedelta}


#    def _tensor_flow_model(self):
#        from keras.models import Sequential
#        from keras.layers import Dense, Dropout, Activation, Flatten
#        classifier = Sequential()
#        classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
#            activation = 'relu', input_dim = 30))
#        classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
#            activation = 'relu'))
#        classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
#            activation = 'sigmoid'))
#        classifier.compile(optimizer = 'adam',
#                           loss = 'binary_crossentropy',
#                           metrics = ['accuracy'])
#        return classifier
#        model = Sequential()
#        model.add(Activation('relu'))
#        model.add(Activation('relu'))
#        model.add(Dropout(0.25))
#        model.add(Flatten())
#        for layer_size in self.parameters['dense_layer_sizes']:
#            model.add(Dense(layer_size))
#            model.add(Activation('relu'))
#        model.add(Dropout(0.5))
#        model.add(Dense(2))
#        model.add(Activation('softmax'))
#        model.compile(loss = 'categorical_crossentropy',
#                      optimizer = 'adadelta',
#                      metrics = ['accuracy'])
#        return model



def create_torch_model(technique: Technique, parameters: dict):
    algorithm = None
    return algorithm


def create_stan_model(technique: Technique, parameters: dict):
    algorithm = None
    return algorithm

