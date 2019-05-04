"""
The Tube class, which contains a single "test tube" of methods to be applied
in a machine learning experiment.
"""
from dataclasses import dataclass

from ml_funnel.methods import Methods

@dataclass
class Tube(object):
    """
    Class containing single test tube of methods.
    """
    steps : object = None
    scaler : object = None
    splitter : object = None
    encoder : object = None
    interactor : object = None
    splicer : object = None
    sampler : object = None
    selector : object = None
    custom : object = None
    model : object = None
    plotter : object = None
    settings : object = None

    def __post_init__(self):
        default_order = ['scaler', 'splitter', 'encoder', 'interactor',
                         'splicer', 'sampler', 'selector', 'custom', 'model',
                         'plotter']
        if not self.steps:
            self.steps = default_order
        return self

    def _scalers(self):
        if self.scaler:
            self.data.x[self.data.num_cols] = (
                    self.scaler.apply(self.data.x[self.data.num_cols]))
        return self

    def _splitter(self):
        if self.splitter:
            self.data = self.splitter.apply(self.data)
        if self.use_val_set:
            (self.data.x_train, self.data.y_train, self.data.x_test,
             self.data.y_test) = self.data['val', True]
        elif self.use_full_set:
            (self.data.x_train, self.data.y_train, self.data.x_test,
             self.data.y_test) = self.data[ 'full', True]
        else:
            (self.data.x_train, self.data.y_train, self.data.x_test,
             self.data.y_test) = self.data['test', True]
        return self

    def _encoders(self):
        if self.encoder.name != 'none':
            self.encoder.fit(self.data.x_train, self.data.y_train)
            self.data.x_train = (
                    self.encoder.transform(self.data.x_train.reset_index(
                    drop = True)))
            self.data.x_test = (
                    self.encoder.transform(self.data.x_test.reset_index(
                    drop = True)))
            self.data.x = (self.encoder.transform(self.data.x.reset_index(
                    drop = True)))
        return self

    def _interactors(self):
        if self.interactor.name != 'none':
            self.interactor.fit(self.data.x_train, self.data.y_train)
            self.data.x_train = (
                    self.interactor.transform(self.data.x_train.reset_index(
                    drop = True)))
            self.data.x_test = (
                    self.interactor.transform(self.data.x_test.reset_index(
                    drop = True)))
            self.data.x = (
                    self.interactor.transform(self.data.x.reset_index(
                    drop = True)))
        return self

    def _splicers(self):
        if self.splicer.name != 'none':
            self.data.x_train = self.splicer.transform(self.data.x_train)
            self.data.x_test = self.splicer.transform(self.data.x_test)
        return self

    def _samplers(self):
        if self.sampler.name != 'none':
            self.data.x_train, self.data.y_train = (
                    self.sampler.method.fit_resample(self.data.x_train,
                                                     self.data.y_train))
        return self

    def _customs(self):
        if self.custom.name != 'none':
            self.custom.apply(self.data.x_train, self.data.y_train)
        return self

    def _selectors(self):
        if self.selector.name != 'none':
            self.selector.fit(self.data.x_train, self.data.y_train)
            self.data.x_train = self.selector.transform(self.data.x_train)
        return self

    def _models(self):
        if self.model.name != 'none':
            if self.model.hyperparameter_search:
                self.model.search(self.data.x_train, self.data.y_train)
                self.model.method = self.model.best
            else:
                self.model.method.fit(self.data.x_train, self.data.y_train)
        return self

    def _plotters(self):
        if self.plotter:
            self.plotter.apply(data = self.data,
                               model = self.model,
                               tube_num = self.tube_num,
                               splicer = self.splicer)
        return self

    def apply(self, data, tube_num = 1, use_full_set = False,
              use_val_set = False):
        """
        Applies the Tube methods to the passed data. If use_full_set is
        selected, methods are applied to entire x and y. If use_val_set
        is selected, methods are applied to x_val and y_val (which are
        created by the data splitter according to user specifications).
        Otherwise, x_test and y_test are used. With either the test or val
        sets selected, x_train and y_train are used for training. With the
        full set, x and y are used for both training and testing (which will
        ordinarily lead to a much higher level of accuracy). The full set
        option should, accordingly, not be used for testing the model's
        performance.
        Scaling is performed on the entire x data regardless of the option
        selected because it does not create exogenity issues for the model.
        """
        self.data = data
        self.tube_num = tube_num
        self.use_full_set = use_full_set
        self.use_val_set = use_val_set
        for step in self.steps:
            step_name = '_' + step
            method = getattr(self, step_name)
            method()
        return self