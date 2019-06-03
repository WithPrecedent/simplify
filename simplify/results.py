
from dataclasses import dataclass
import re

import pandas as pd
import sklearn.metrics as met

from .steps.custom import Custom
from .steps.encoder import Encoder
from .steps.interactor import Interactor
from .steps.model import Model
from .steps.plotter import Plotter
from .recipe import Recipe
from .steps.sampler import Sampler
from .steps.scaler import Scaler
from .steps.selector import Selector
from .steps.splicer import Splicer
from .steps.splitter import Splitter
from .steps.step import Step


@dataclass
class Results(Step):
    """Class for storing machine learning experiment results.

    Results  creates and stores a results table and other general
    scorers/metrics for machine learning based upon the type of model used in
    the siMpLify package. Users can manually add metrics not already included
    in the metrics dictionary by passing them to Results.include.
    """
    name : str = 'results'
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self, sections = ['general',
                                                            'recipes'])
        self.step_columns = ['recipe_number', 'step_order', 'scaler',
                             'splitter', 'encoder', 'interactor', 'splicer',
                             'sampler', 'custom', 'selector', 'model', 'seed',
                             'validation_set']
        self.columns = self.step_columns
        self._set_options()
        self.columns.extend(self._listify(self.metrics))
        self.table = pd.DataFrame(columns = self.columns)
        return self

    def _set_options(self):
        """
        Sets default metrics for scores dataframe based upon the type of
        model used.
        """
        self.spec_metrics = {'fbeta' : {'beta' : 1},
                             'f1_weighted' : {'average' : 'weighted'},
                             'precision_weighted' : {'average' : 'weighted'},
                             'recall_weighted' : {'average' : 'weighted'}}
        self.neg_metrics = ['brier_loss_score', 'neg_log_loss', 'zero_one']
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
            self.prob_options = {'brier_score_loss' : met.brier_score_loss}
            self.score_options = {'roc_auc' :  met.roc_auc_score}
        elif self.model_type in ['regressor']:
            self.options = {
                    'explained_variance' : met.explained_variance_score,
                    'max_error' : met.max_error,
                    'absolute_error' : met.absolute_error,
                    'mse' : met.mean_squared_error,
                    'msle' : met.mean_squared_log_error,
                    'mae' : met.median_absolute_error,
                    'r2' : met.r2_score}
            self.prob_options = {}
            self.score_options = {}
        elif self.model_type in ['clusterer']:
            self.options = {
                    'adjusted_mutual_info' : met.adjusted_mutual_info_score,
                    'adjusted_rand' : met.adjusted_rand_score,
                    'calinski' : met.calinski_harabasz_score,
                    'davies' : met.davies_bouldin_score,
                    'completeness' : met.completeness_score,
                    'contingency_matrix' : met.cluster.contingency_matrix,
                    'fowlkes' : met.fowlkes_mallows_score,
                    'h_completness' : met.homogeneity_completeness_v_measure,
                    'homogeniety' : met.homogeneity_score,
                    'mutual_info' : met.mutual_info_score,
                    'norm_mutual_info' : met.normalized_mutual_info_score,
                    'silhouette' : met.silhouette_score,
                    'v_measure' : met.v_measure_score}
            self.prob_options = {}
            self.score_options = {}
        self.options.update(self.prob_options)
        self.options.update(self.score_options)
        return self

    def _parse_step(step, return_cols = False):
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
        model = Model(self._parse_step(row['model']))
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

    def include(self, name, metric, special_type = None, special_params = None,
                negative_metric = False):
        """
        Allows user to manually add a metric to the scores dataframe.
        """
        self.options.update({name : metric})
        if special_type in ['probability']:
            self.prob_options.update({name : metric})
        elif special_type in ['scorer']:
            self.score_options.update({name : metric})
        if special_params:
           self.spec_metrics.update({name : special_params})
        if negative_metric:
           self.spec_metrics.append[name]
        self._set_options()
        self.table = pd.DataFrame(columns = self.columns)
        return self

    def load(self, import_folder = None, file_name = 'results_table',
             file_format = 'csv', import_path = '', encoding = 'windows-1252',
             message = 'Importing results'):
        """
        Imports results scores file from disc. This method is used if the user
        wants to reconstruct recipes or cookbooks from past experiments
        """
        if self.verbose:
            print(message)
        if import_path:
            results_path = import_path
        elif self.results_path:
            results_path = self.results_path
        else:
            results_path = self.filer.make_path(folder = import_folder,
                                                file_name = file_name,
                                                file_type = file_format)
        self.table = self.filer.load(results_path,
                                     encoding = encoding)
        return self

    def save(self, export_path = None, file_name = 'results_table',
             file_format = 'csv', encoding = 'windows-1252',
             float_format = '%.4f', message = 'Exporting results'):
        """
        Exports results scores to disc.
        """
        if not export_path:
            export_path = self.filer.results_folder
            export_path = self.filer.make_path(folder = export_path,
                                               name = file_name,
                                               file_type = file_format)
        self.table.to_csv(export_path,
                          encoding = encoding,
                          float_format = float_format)
        self.filer.save(df = self.table,
                        export_path = export_path,
                        encoding = encoding,
                        float_format = float_format,
                        message = message)
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