"""
Results is a class which creates and stores a results table and other
general scorers/metrics for machine learning based upon the type of model
used in the siMpLify package. Users can manually add metrics not already
included in the metrics dictionary by passing them to add_metric.
"""
from dataclasses import dataclass
import pandas as pd
import re
import sklearn.metrics as met

#import eli5

from custom import Custom
from encoder import Encoder
from filer import Filer
from interactor import Interactor
from model import Model
from plotter import Plotter
from recipe import Recipe
from sampler import Sampler
from scaler import Scaler
from selector import Selector
from splicer import Splicer
from splitter import Splitter
from step import Step


@dataclass
class Results(Step):
    """
    Class for storing machine learning experiment results.
    """
    settings : object

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self, sections = ['results'])
        self.step_columns = ['recipe_number', 'step_order', 'predictors',
                             'scaler', 'splitter', 'encoder', 'interactor',
                             'sampler', 'custom', 'selector', 'model', 'seed',
                             'validation_set']
        self.columns = self.step_columns
        self._set_metrics()
        self.columns.extend(self.metrics)
        self.table = pd.DataFrame(columns = self.columns)
        return self

    @staticmethod
    def _check_none(step):
        """
        Checks if metric listed is either 'none' or 'all.' Otherwise, it
        returns the name of the algorithm selected.
        """
        if step.name in ['none', 'all']:
            return step.name
        elif not step.name:
            return 'none'
        else:
            return step.algorithm

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

    def _parse_step(self, step, return_cols = False):
        if step == 'none':
            name = 'none'
            params = {}
        else:
            name = re.search('^\D*.?(?=\, parameters)', step)[0]
            params = re.search('\{.*?\}', step)[0]
        if return_cols:
            step = re.sub('\{.*?\}', '')
            cols = re.search('\[.*', step)[0]
            return name, params, cols
        else:
            return name, params

    def _parse_result(self, row):
        model = Mode(self._parse_step(row['model']))
        recipe = Recipe(row['recipe_number'],
                        order = row['step_order'].split(),
                        scaler = Scaler(self._parse_step(row['scaler'],
                                                    return_columns = True)),
                        splitter = Splitter(self._parse_step(row['splitter'])),
                        encoder = Encoder(self._parse_step(row['encoder'],
                                                    return_cols = True)),
                        interactor = Interactor(self._parse_step(
                                row['interactor'], return_cols = True)),
                        splicer = Splicer(self._parse_step(row['splicer'])),
                        sampler = Sampler(self._parse_step(row['sampler'])),
                        custom = Custom(self._parse_step(row['custom'])),
                        selector = Selector(self._parse_step(row['selector'])),
                        model = model,
                        plotter = Plotter(self._parse_step(row['plotter']),
                                          model),
                        settings = self.settings)
        return recipe

    def add_metric(self, name, metric):
        """
        Allows user to manually add a metric to the results table.
        """
        self.metric_dict.update({name : metric})
        return self

    def add_result(self, recipe, use_val_set = False):
        """
        Adds the results of a single recipe application to the results table.
        """
        self.predictions = recipe.model.algorithm.predict(recipe.data.x_test)
        self.pred_probs = recipe.model.algorithm.predict_proba(
                recipe.data.x_test)
        new_row = pd.Series(index = self.columns)
        recipe_cols = {'recipe_number' : recipe.number,
                       'step_order' : self.order,
                       'predictors' : recipe.splicer,
                       'scaler' : recipe.scaler,
                       'splitter' : recipe.splitter,
                       'encoder' : recipe.encoder,
                       'interactor' : recipe.interactor,
                       'sampler' : recipe.sampler,
                       'custom' : recipe.custom,
                       'selector' : recipe.selector,
                       'model' : recipe.model,
                       'seed' : recipe.model.seed,
                       'validation_set' : use_val_set}
        print(recipe)
        for key, value in recipe_cols.items():
            if key in ['recipe_number', 'step_order', 'validation_set',
                       'seed']:
                new_row[key] = value
            else:
                name = value.name
                params = value.params
                if hasattr(value, 'columns'):
                    cols = value.columns
                else:
                    cols = ['all']
                if name in ['none']:
                    new_row[key] = name
                else:
                    new_row[key] = (
                        f'{name}, parameters = {params}, columns = {cols}')
        for key, value in self.options.items():
            if key in self.metrics:
                if key in self.prob_options:
                    params = {'y_true' : recipe.data.y_test,
                              'y_prob' : self.pred_probs[:, 1]}
                elif key in self.score_options:
                    params = {'y_true' : recipe.data.y_test,
                              'y_score' : self.pred_probs[:, 1]}
                else:
                    params = {'y_true' : recipe.data.y_test,
                              'y_pred' : self.predictions}
                if key in self.spec_metrics:
                    params.update({key : self.spec_metrics[key]})
                result = value(**params)
                if key in self.neg_metrics:
                    result = -1 * result
                new_row[key] = result
        self.table.loc[len(self.table)] = new_row
        self._other_results(recipe)
        return self

    def _other_results(self, recipe):
        """
        Creates attributes storing other common metrics and tables.
        """
        self.confusion = met.confusion_matrix(recipe.data.y_test,
                                              self.predictions)
        self.class_report = met.classification_report(recipe.data.y_test,
                                                      self.predictions)
        self.feature_list = list(recipe.data.x_test.columns)
        self.feature_import = pd.Series(
                data = recipe.model.algorithm.feature_importances_,
                index = self.feature_list)
        self.feature_import.sort_values(ascending = False,
                                        inplace = True)
        if self.verbose:
            print('These are the results using the', recipe.model.name,
                  'model')
            print('Testing', recipe.splicer.name, 'predictors')
            print('Confusion Matrix:')
            print(self.confusion)
            print('Classification Report:')
            print(self.class_report)
        return self

    def get_best_recipe(self):
        recipe_row = self.table[self.metrics[0]].argmax()
        recipe = self._parse_result(recipe_row)
        return recipe

    def get_recipe(self, recipe_number = None, scorer = None):
        if recipe_number:
            recipe_row = self.table.iloc[recipe_number - 1]
        elif scorer:
            recipe_row = self.table[scorer].argmax()
        recipe = self._parse_result(recipe_row)
        return recipe

    def get_all_recipes(self):
        recipes = []
        for row in self.table.iterrows():
            recipes.append(self._parse_result(self.table[row]))
        return recipes