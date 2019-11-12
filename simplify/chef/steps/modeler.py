"""
.. module:: modeler
:synopsis: applies machine learning and statistical models to data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleComposer
from simplify.core.technique import SimpleDesign


@dataclass
class Modeler(SimpleComposer):
    """Splits data into training, testing, and/or validation datasets.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'modeler'

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Private Methods """

    def _add_gpu_techniques_classify(self) -> None:
        self.options.update({
            'forest_inference': SimpleDesign(
                name = 'forest_inference',
                module = 'cuml',
                algorithm = 'ForestInference'),
            'random_forest': SimpleDesign(
                name = 'random_forest',
                module = 'cuml',
                algorithm = 'RandomForestClassifier'),
            'logit': SimpleDesign(
                name = 'logit',
                module = 'cuml',
                algorithm = 'LogisticRegression')})
        return self

    def _add_gpu_techniques_cluster(self) -> None:
        self.options.update({
            'dbscan': SimpleDesign(
                name = 'dbscan',
                module = 'cuml',
                algorithm = 'DBScan'),
            'kmeans': SimpleDesign(
                name = 'kmeans',
                module = 'cuml',
                algorithm = 'KMeans')})
        return self

    def _add_gpu_techniques_regress(self) -> None:
        self.options.update({
            'lasso': SimpleDesign(
                name = 'lasso',
                module = 'cuml',
                algorithm = 'Lasso'),
            'ols': SimpleDesign(
                name = 'ols',
                module = 'cuml',
                algorithm = 'LinearRegression'),
            'ridge': SimpleDesign(
                name = 'ridge',
                module = 'cuml',
                algorithm = 'RidgeRegression')})
        return self

    def _draft_classify(self) -> None:
        self.options = {
            'adaboost': SimpleDesign(
                name = 'adaboost',
                module = 'sklearn.ensemble',
                algorithm = 'AdaBoostClassifier'),
            'baseline_classifier': SimpleDesign(
                name = 'baseline_classifier',
                module = 'sklearn.dummy',
                algorithm = 'DummyClassifier',
                required = {'strategy': 'most_frequent'}),
            'logit': SimpleDesign(
                name = 'logit',
                module = 'sklearn.linear_model',
                algorithm = 'LogisticRegression'),
            'random_forest': SimpleDesign(
                name = 'random_forest',
                module = 'sklearn.ensemble',
                algorithm = 'RandomForestClassifier'),
            'svm_linear': SimpleDesign(
                name = 'svm_linear',
                module = 'sklearn.svm',
                algorithm = 'SVC',
                required = {'kernel': 'linear', 'probability': True}),
            'svm_poly': SimpleDesign(
                name = 'svm_poly',
                module = 'sklearn.svm',
                algorithm = 'SVC',
                required = {'kernel': 'poly', 'probability': True}),
            'svm_rbf': SimpleDesign(
                name = 'svm_rbf',
                module = 'sklearn.svm',
                algorithm = 'SVC',
                required = {'kernel': 'rbf', 'probability': True}),
            'svm_sigmoid': SimpleDesign(
                name = 'svm_sigmoid ',
                module = 'sklearn.svm',
                algorithm = 'SVC',
                required = {'kernel': 'sigmoid', 'probability': True}),
            'tensorflow': SimpleDesign(
                name = 'tensorflow',
                module = 'tensorflow',
                algorithm = None,
                default = {
                    'batch_size': 10,
                    'epochs': 2}),
            'xgboost': SimpleDesign(
                name = 'xgboost',
                module = 'xgboost',
                algorithm = 'XGBClassifier',
                data_dependent = 'scale_pos_weight')}
        return self

    def _draft_cluster(self) -> None:
        self.options = {
            'affinity': SimpleDesign(
                name = 'affinity',
                module = 'sklearn.cluster',
                algorithm = 'AffinityPropagation'),
            'agglomerative': SimpleDesign(
                name = 'agglomerative',
                module = 'sklearn.cluster',
                algorithm = 'AgglomerativeClustering'),
            'birch': SimpleDesign(
                name = 'birch',
                module = 'sklearn.cluster',
                algorithm = 'Birch'),
            'dbscan': SimpleDesign(
                name = 'dbscan',
                module = 'sklearn.cluster',
                algorithm = 'DBSCAN'),
            'kmeans': SimpleDesign(
                name = 'kmeans',
                module = 'sklearn.cluster',
                algorithm = 'KMeans'),
            'mean_shift': SimpleDesign(
                name = 'mean_shift',
                module = 'sklearn.cluster',
                algorithm = 'MeanShift'),
            'spectral': SimpleDesign(
                name = 'spectral',
                module = 'sklearn.cluster',
                algorithm = 'SpectralClustering'),
            'svm_linear': SimpleDesign(
                name = 'svm_linear',
                module = 'sklearn.cluster',
                algorithm = 'OneClassSVM'),
            'svm_poly': SimpleDesign(
                name = 'svm_poly',
                module = 'sklearn.cluster',
                algorithm = 'OneClassSVM'),
            'svm_rbf': SimpleDesign(
                name = 'svm_rbf',
                module = 'sklearn.cluster',
                algorithm = 'OneClassSVM,'),
            'svm_sigmoid': SimpleDesign(
                name = 'svm_sigmoid',
                module = 'sklearn.cluster',
                algorithm = 'OneClassSVM')}
        return self

    def _draft_regress(self) -> None:
        self.options = {
            'adaboost': SimpleDesign(
                name = 'adaboost',
                module = 'sklearn.ensemble',
                algorithm = 'AdaBoostRegressor'),
            'baseline_regressor': SimpleDesign(
                name = 'baseline_regressor',
                module = 'sklearn.dummy',
                algorithm = 'DummyRegressor',
                required = {'strategy': 'mean'}),
            'bayes_ridge': SimpleDesign(
                name = 'bayes_ridge',
                module = 'sklearn.linear_model',
                algorithm = 'BayesianRidge'),
            'lasso': SimpleDesign(
                name = 'lasso',
                module = 'sklearn.linear_model',
                algorithm = 'Lasso'),
            'lasso_lars': SimpleDesign(
                name = 'lasso_lars',
                module = 'sklearn.linear_model',
                algorithm = 'LassoLars'),
            'ols': SimpleDesign(
                name = 'ols',
                module = 'sklearn.linear_model',
                algorithm = 'LinearRegression'),
            'random_forest': SimpleDesign(
                name = 'random_forest',
                module = 'sklearn.ensemble',
                algorithm = 'RandomForestRegressor'),
            'ridge': SimpleDesign(
                name = 'ridge',
                module = 'sklearn.linear_model',
                algorithm = 'Ridge'),
            'svm_linear': SimpleDesign(
                name = 'svm_linear',
                module = 'sklearn.svm',
                algorithm = 'SVC',
                required = {'kernel': 'linear', 'probability': True}),
            'svm_poly': SimpleDesign(
                name = 'svm_poly',
                module = 'sklearn.svm',
                algorithm = 'SVC',
                required = {'kernel': 'poly', 'probability': True}),
            'svm_rbf': SimpleDesign(
                name = 'svm_rbf',
                module = 'sklearn.svm',
                algorithm = 'SVC',
                required = {'kernel': 'rbf', 'probability': True}),
            'svm_sigmoid': SimpleDesign(
                name = 'svm_sigmoid ',
                module = 'sklearn.svm',
                algorithm = 'SVC',
                required = {'kernel': 'sigmoid', 'probability': True}),
            'xgboost': SimpleDesign(
                name = 'xgboost',
                module = 'xgboost',
                algorithm = 'XGBRegressor',
                data_dependent = 'scale_pos_weight')}
        return self

    def _get_conditional(self,
            technique: str,
            parameters: Dict[str, Any]) -> None:
        
        if technique in ['xgboost'] and self.gpu:
            parameters.update({'tree_method': 'gpu_exact'})
        elif technique in ['tensorflow']:
            algorithm = create_tensorflow_model(
                technique = technique,
                parameters = parameters)
        return parameters

    """ Core siMpLify Methods """

    def draft(self) -> None:
        super().draft()
        getattr(self, ''.join('_draft_', self.model_type))()
        if self.gpu:
            getattr(self, ''.join('_add_gpu_techniques_', self.model_type))()
        return self


def create_tensorflow_model(technique: Technique, parameters: dict) -> None:
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



def create_torch_model(technique: Technique, parameters: dict) -> None:
    algorithm = None
    return algorithm


def create_stan_model(technique: Technique, parameters: dict) -> None:
    algorithm = None
    return algorithm

