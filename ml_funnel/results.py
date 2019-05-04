"""
Class and methods for ml_funnel which creates a results table and other
general scorers/metrics for machine learning based upon the type of model
used. Users can manually add metrics not already included in the metrics
dictionary by passing them to add_metric.
"""
from dataclasses import dataclass
import pandas as pd
import sklearn.metrics as met

#import eli5

from ml_funnel.methods import Methods

@dataclass
class Results(Methods):
    """
    Class for storing machine learning experiment results.
    """
    settings : object

    def __post_init__(self):
        super().__post_init__()
        self.settings.simplify(class_instance = self, sections = ['results'])
        self.step_columns = ['tube_number', 'step_order', 'predictors',
                             'scaler', 'splitter', 'splicer', 'encoder',
                             'interactor', 'sampler', 'custom', 'selector',
                             'model', 'seed', 'validation_set']
        self.columns = self.step_columns
        self._set_metrics()
        self.columns.extend(self.metrics)
        self.table = pd.DataFrame(columns = self.columns)
        return self

    @staticmethod
    def _check_none(step):
        """
        Checks if metric listed is either 'none' or 'all.' Otherwise, it
        returns the name of the method selected.
        """
        if step.name in ['none', 'all']:
            return step.name
        elif not step.name:
            return 'none'
        else:
            return step.method

    def _set_metrics(self):
        """
        Sets default metrics for the results table based upon the type of
        model used. For metrics were lower values indicate better results,
        the returned result is the 1 - the metric value. For example,
        'neg_log_loss' = 1 - the sklearn log_loss metric.
        """
        self.spec_metrics = {'fbeta' : {'beta' : 1},
                             'f1_weighted' : {'average' : 'weighted'},
                             'precision_weighted' : {'average' : 'weighted'},
                             'recall_weighted' : {'average' : 'weighted'}}
        self.neg_metrics = ['brier_loss_score', 'neg_log_loss']
        if self.model_type in ['classifier']:
            self.options = {
                    'accuracy' : met.accuracy_score,
                    'balanced_accuracy' : met.balanced_accuracy_score,
                    'f1' : met.f1_score,
                    'f1_weighted' : met.f1_score,
                    'fbeta' : met.fbeta_score,
                    'hamming' : met.hamming_loss,
                    'jaccard' : met.jaccard_similarity_score,
                    'matthews_corrcoef' : met.matthews_corrcoef,
                    'neg_log_loss' :  met.log_loss,
                    'precision' :  met.precision_score,
                    'precision_weighted' :  met.precision_score,
                    'recall' :  met.recall_score,
                    'recall_weighted' :  met.recall_score,

                    'zero_one' : met.zero_one_loss}
            self.prob_options = {
                    'brier_score_loss' : met.brier_score_loss}
            self.score_options = {
                    'roc_auc' :  met.roc_auc_score}
        elif self.model_type in ['regressor']:
            self.options = {}
            self.prob_options = {}
            self.score_options = {}
        elif self.model_type in ['grouper']:
            self.options = {}
            self.prob_options = {}
            self.score_options = {}
        self.options.update(self.prob_options)
        self.options.update(self.score_options)
        return self

    def add_metric(self, name, metric):
        """
        Allows user to manually add a metric to the results table.
        """
        self.metric_dict.update({name : metric})
        return self

    def add_result(self, tube, use_val_set = False):
        """
        Adds the results of a single tube application to the results table.
        """
        self.predictions = tube.model.method.predict(tube.data.x_test)
        self.pred_probs = tube.model.method.predict_proba(tube.data.x_test)
        new_row = pd.Series(index = self.columns)
        tube_cols = {'tube_number' : tube.tube_num,
                     'step_order' : self.steps,
                     'predictors' : tube.splicer,
                     'scaler' : tube.scaler,
                     'splitter' : tube.splitter,
                     'encoder' : tube.encoder,
                     'interactor' : tube.interactor,
                     'splicer' : tube.splicer,
                     'sampler' : tube.sampler,
                     'custom' : tube.custom,
                     'selector' : tube.selector,
                     'model' : tube.model,
                     'seed' : tube.model.seed,
                     'validation_set' : use_val_set}
        for key, value in tube_cols.items():
            new_row[key] = value
        for key, value in self.options.items():
            if key in self.metrics:
                if key in self.prob_options:
                    params = {'y_true' : tube.data.y_test,
                              'y_prob' : self.pred_probs[:, 1]}
                elif key in self.score_options:
                    params = {'y_true' : tube.data.y_test,
                              'y_score' : self.pred_probs[:, 1]}
                else:
                    params = {'y_true' : tube.data.y_test,
                              'y_pred' : self.predictions}
                if key in self.spec_metrics:
                    params.update({key : self.spec_metrics[key]})
                result = value(**params)
                if key in self.neg_metrics:
                    result = -1 * result
                new_row[key] = result
        self.table.loc[len(self.table)] = new_row
        self._other_results(tube)
        return self

    def _other_results(self, tube):
        """
        Creates attributes storing other common metrics and tables.
        """
        self.confusion = met.confusion_matrix(tube.data.y_test,
                                              self.predictions)
        self.class_report = met.classification_report(tube.data.y_test,
                                                      self.predictions)
        self.feature_list = list(tube.data.x_test.columns)
        self.feature_import = pd.Series(
                data = tube.model.method.feature_importances_,
                index = self.feature_list)
        self.feature_import.sort_values(ascending = False,
                                        inplace = True)
        if self.verbose:
            print('These are the results using the', tube.model.name,
                  'model')
            print('Testing', tube.splicer.name, 'predictors')
            print('Confusion Matrix:')
            print(self.confusion)
            print('Classification Report:')
            print(self.class_report)
        return self