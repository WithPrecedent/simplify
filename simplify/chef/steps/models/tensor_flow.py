
from dataclasses import dataclass

import tensorflow as tf

from simplify.core.base import SimpleTechnique


@dataclass
class TFModel(SimpleTechnique):
    """Applies machine learning algorithms based upon user selections."""
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'model'

    def __post_init__(self):
        super().__post_init__()
        return self

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
#        self.type_interface = {'boolean' : tf.bool,
#                               'float' : tf.float16,
#                               'integer' : tf.int8,
#                               'string' : object,
#                               'categorical' : CategoricalDtype,
#                               'list' : list,
#                               'datetime' : datetime64,
#                               'timedelta' : timedelta}

    def draft(self):
        self.model_parameters = {'build_fn' : self._tensor_flow_model,
                                 'batch_size' : 10,
                                 'epochs' : 2}


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
