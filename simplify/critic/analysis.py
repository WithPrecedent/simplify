"""
.. module:: analysis
  :synopsis: core classes for Critic subpackage.
  :author: Corey Rayburn Yung
  :copyright: 2019
  :license: CC-BY-NC-4.0

This is the primary control file for evaluating, summarizing, and analyzing 
data, as well as machine learning and other statistical models.

Contents:
    Analysis: primary class for model evaluation and preparing reports about
        that evaluation and data.
    Review: class for storing metrics, evaluations, and reports related to data 
        and models.
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.base import SimpleManager, SimplePlan


@dataclass
class Analysis(SimpleManager):
    """Summarizes, evaluates, and creates visualizations for data and data
    analysis from the siMpLify Harvest and Cookbook.

    Args:
        steps: an ordered list of step names to be completed. This argument
            should only be passed if the user whiches to override the steps
            listed in 'idea.configuration'. 
        name (str): designates the name of the class which should be identical 
            to the section of the idea configuration with relevant settings.
        auto_finalize (bool): whether to call the 'finalize' method when the 
            class is instanced.
        auto_produce (bool): whether to call the 'produce' method when the class 
            is instanced.
    """
    steps : object = None
    name : str = 'analysis'
    auto_finalize : bool = True
    auto_produce : bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """
    
    def _check_best(self, recipe):
        """Checks if the current recipe is better than the current best recipe
        based upon the primary scoring metric.

        Args:
            recipe: an instance of Recipe to be tested versus the current best
                recipe stored in the 'best_recipe' attribute.
        """
        if not hasattr(self, 'best_recipe') or self.best_recipe is None:
            self.best_recipe = recipe
            self.best_recipe_score = self.report.loc[
                    self.report.index[-1],
                    self.listify(self.metrics)[0]]
        elif (self.report.loc[
                self.report.index[-1],
                self.listify(self.metrics)[0]] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.report.loc[
                    self.report.index[-1],
                    self.listify(self.metrics)[0]]
        return self

    """ Public Tool Methods """
    
    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  self.listify(self.metrics)[0], 'metric with a score of',
                  f'{self.best_recipe_score : 4.4f}', 'is:')
            for technique in getattr(self,
                    self.plan_iterable).best_recipe.techniques:
                print(technique.capitalize(), ':',
                      getattr(getattr(self, self.plan_iterable).best_recipe,
                              technique).technique)
        return

    """ Core siMpLify methods """
    
    def draft(self):
        """Sets default options for the Critic's analysis."""
        self.options = {'summarizer' : Summarize,
                        'predictor' : Predict,
                        'scorer' : Score,
                        'evaluator' : Evaluate,
                        'review' : Review}
        # Sets check methods to run.
        self.checks = ['steps']
        # Locks 'step' attribute at 'critic' for conform methods in package.
        self.step = 'critic'
        # Sets 'manager_type' so that proper parent methods are used.
        self.manager_type = 'serial'
        # Sets plan-related attributes to allow use of parent methods.
        self.plan_class = Review
        self.plan_iterable = 'review'
        return self
    
    def produce(self, ingredients = None, recipes = None):
        """Evaluates recipe with various tools and finalizes report.
        
        Args: 
            ingredients (Ingredients): an instance or subclass instance of 
                Ingredients. 
            recipes (list or Recipe): a Recipe or a list of Recipes.
        """
        if recipes:
            self.recipes = recipes
        super().produce(plans = recipes)
        return self

    
@dataclass
class Summarize(SimplePlan):
    """Stores and exports a DataFrame of summary data for pandas DataFrame.

    Summary is more inclusive than pandas.describe() and includes
    boolean and numerical columns by default. It is also extensible: more
    metrics can easily be added to the report DataFrame.

    Args:
        name: a string designating the name of the class which should be
            identical to the section of the idea configuration with relevant
            settings.
        auto_finalize: a boolean value that sets whether the finalize method is
            automatically called when the class is instanced.
        auto_produce: sets whether to automatically call the 'produce' method
            when the class is instanced.
    """
    steps : object = None
    name : str = 'summarizer'
    auto_finalize : bool = True
    auto_produce : bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Core siMpLify Methods """
    
    def draft(self):
        """Sets options for Summarize class."""
        self.options = {'datatype' : ['dtype'],
                        'count' : 'count',
                        'min' :'min',
                        'q1' : ['quantile', 0.25],
                        'median' : 'median',
                        'q3' : ['quantile', 0.75],
                        'max' : 'max',
                        'mad' : 'mad',
                        'mean' : 'mean',
                        'stan_dev' : 'std',
                        'mode' : ['mode', [0]],
                        'sum' : 'sum',
                        'kurtosis' : 'kurtosis',
                        'skew' : 'skew',
                        'variance' : 'var',
                        'stan_error' : 'sem',
                        'unique' : 'nunique'}
        return self
    
    def finalize(self):
        """Prepares columns list for Summary report and initializes report."""
        self.columns = ['variable']
        self.columns.extend(list(self.options.keys()))
        self.statistics = pd.DataFrame(columns = self.columns)
        return self
    
    def produce(self, df = None, transpose = True, file_name = 'data_summary',
                file_format = 'csv'):
        """Creates and exports a DataFrame of common summary data using the
        Summary class.

        Args:
            df: a pandas DataFrame.
            transpose: boolean value indicating whether the df columns should
                be listed horizontally (True) or vertically (False) in report.
            file_name: string containing name of file to be exported.
            file_format: string of file extension from Depot.extensions.
        """
        """Completes report with data from df.

        Args:
            df: pandas DataFrame.
            transpose: boolean value indicating whether the df columns should be
                listed horizontally (True) or vertically (False) in report.
        """
        self.file_name = file_name
        for column in df.columns:
            row = pd.Series(index = self.columns)
            row['variable'] = column
            if df[column].dtype == bool:
                df[column] = df[column].astype(int)
            if df[column].dtype.kind in 'bifcu':
                for key, value in self.options.items():
                    if isinstance(value, str):
                        row[key] = getattr(df[column], value)()
                    elif isinstance(value, list):
                        if len(value) < 2:
                            row[key] = getattr(df[column], value[0])
                        elif isinstance(value[1], list):
                            row[key] = getattr(df[column],
                               value[0])()[value[1]]
                        else:
                            row[key] = getattr(df[column],
                               value[0])(value[1])
            self.statistics.loc[len(self.statistics)] = row
        if not transpose:
            self.statistics = self.statistics.transpose()
            self.df_header = False
            self.df_index = True
        else:
            self.df_header = True
            self.df_index = False
        return self
    
@dataclass
class Predict(SimplePlan):
    """Core class for making predictions based upon machine learning models.

    """
    steps : object = None
    name : str = 'predictor'
    auto_finalize : bool = True
    
    def __post_init__(self):
        """Sets up the core attributes of an Evaluator instance."""
        super().__post_init__()
        return self        


    def _predict_outcomes(self, recipe):
        """Makes predictions from fitted model.
        
        Args: 
            recipe(Recipe): instance of Recipe with a fitted model.
        
        Returns:
            Series with predictions from fitted model on test data.
        """
        if hasattr(self.recipe.model.algorithm, 'predict'):
            return self.recipe.model.algorithm.predict(
                self.recipe.ingredients.x_test)
        else:
            if self.verbose:
                print('predict method does not exist for', 
                    self.recipe.model.technique.name)
            return None
        
    def _predict_probabilities(self, recipe):
        """Makes predictions from fitted model.
        
        Args: 
            recipe(Recipe): instance of Recipe with a fitted model.
        
        Returns:
            Series with predicted probabilities from fitted model on test data.
        """
        if hasattr(self.recipe.model.algorithm, 'predict_proba'):
            return self.recipe.model.algorithm.predict_proba(
                    self.recipe.ingredients.x_test)
        else:
            if self.verbose:
                print('predict_proba method does not exist for', 
                    self.recipe.model.technique.name)
            return None
    
    def draft(self):
        self.options = {'outcomes' : self._predict_outcomes,
                        'probabilities' : self._predict_probabilities}
        return self
    
    def produce(self, recipe):
        for step in self.steps:
            setattr(self, 'predicted_' + step, 
                    self.options[step](recipe = recipe))         
        return self
       
@dataclass
class Score(SimplePlan):
    """Core class for evaluating the results of data analysis produceed by
    the siMpLify Cookbook.

    """
    steps : object = None
    name : str = 'scorer'
    auto_finalize : bool = True
    
    def __post_init__(self):
        """Sets up the core attributes of an Evaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        # Local import to conserve memory
        from sklearn import metrics
        self.options = {
            'absolute_error' : metrics.absolute_error,
            'accuracy' : metrics.accuracy_score,
            'adjusted_mutual_info' : metrics.adjusted_mutual_info_score,
            'adjusted_rand' : metrics.adjusted_rand_score,
            'balanced_accuracy' : metrics.balanced_accuracy_score,
            'brier_score_loss' : metrics.brier_score_loss,
            'calinski' : metrics.calinski_harabasz_score,
            'davies' : metrics.davies_bouldin_score,
            'completeness' : metrics.completeness_score,
            'contingency_matrix' : metrics.cluster.contingency_matrix,
            'explained_variance' : metrics.explained_variance_score,
            'f1' : metrics.f1_score,
            'f1_weighted' : metrics.f1_score,
            'fbeta' : metrics.fbeta_score,    
            'fowlkes' : metrics.fowlkes_mallows_score,  
            'hamming' : metrics.hamming_loss,  
            'h_completness' : metrics.homogeneity_completeness_v_measure,
            'homogeniety' : metrics.homogeneity_score,   
            'jaccard' : metrics.jaccard_similarity_score, 
            'mae' : metrics.median_absolute_error,
            'matthews_corrcoef' : metrics.matthews_corrcoef,
            'max_error' : metrics.max_error,        
            'mse' : metrics.mean_squared_error,
            'msle' : metrics.mean_squared_log_error,
            'mutual_info' : metrics.mutual_info_score,
            'neg_log_loss' :  metrics.log_loss,
            'norm_mutual_info' : metrics.normalized_mutual_info_score,
            'precision' :  metrics.precision_score,
            'precision_weighted' :  metrics.precision_score,
            'r2' : metrics.r2_score,
            'recall' :  metrics.recall_score,
            'recall_weighted' :  metrics.recall_score,
            'roc_auc' :  metrics.roc_auc_score,
            'silhouette' : metrics.silhouette_score,
            'v_measure' : metrics.v_measure_score,
            'zero_one' : metrics.zero_one_loss}
        self.prob_options = ['brier_score_loss']
        self.score_options = ['roc_auc']
        self.negative_options = ['brier_loss_score', 'neg_log_loss',
                                 'zero_one']
        self.special_options = {
            'fbeta' : {'beta' : 1},
            'f1_weighted' : {'average' : 'weighted'},
            'precision_weighted' : {'average' : 'weighted'},
            'recall_weighted' : {'average' : 'weighted'}}
        self.checks = ['idea']
        return self  
       
    def edit(self, name, metric, special_type = None, 
             special_parameters = None, negative_metric = False):
        """Allows user to manually add a metric to report."""
        self.options.update({name : metric})
        if special_type in ['probability']:
            self.prob_options.update({name : metric})
        elif special_type in ['scorer']:
            self.score_options.update({name : metric})
        if special_parameters:
           self.special_options.update({name : special_parameters})
        if negative_metric:
           self.negative_options.append[name]
        return self
    
    def produce(self):
        """Prepares the results of a single recipe application to be added to
        the .report dataframe.
        """
        self.result = pd.Series(index = self.columns_list)
        for column, value in self.columns.items():
            if isinstance(getattr(self.recipe, value), CookbookSimpleStep):
                self.result[column] = self._format_step(value)
            else:
                self.result[column] = getattr(self.recipe, value)
        for column, value in self.options.items():
            if column in self.metrics:
                if column in self.prob_options:
                    params = {'y_true' : self.recipe.ingredients.y_test,
                              'y_prob' : self.predicted_probs[:, 1]}
                elif column in self.score_options:
                    params = {'y_true' : self.recipe.ingredients.y_test,
                              'y_score' : self.predicted_probs[:, 1]}
                else:
                    params = {'y_true' : self.recipe.ingredients.y_test,
                              'y_pred' : self.predictions}
                if column in self.special_metrics:
                    params.update({column : self.special_metrics[column]})
                result = value(**params)
                if column in self.negative_metrics:
                    result = -1 * result
                self.result[column] = result
        self.report.loc[len(self.report)] = self.result
        return self    
    
@dataclass
class Evaluate(SimplePlan):
    """Core class for evaluating the results of data analysis produceed by
    the siMpLify Cookbook.

    """
    steps : object = None
    name : str = 'evaluator'
    auto_finalize : bool = True
    
    def __post_init__(self):
        """Sets up the core attributes of an Evaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        # Local import to conserve memory
        from simplify.critic.steps.evaluate import (Eli5Evaluator, 
                                                    ShapEvaluator,
                                                    SkaterEvaluator,
                                                    SklearnEvaluator)
        self.options = {'eli5' : Eli5Evaluator,
                        'shap' : ShapEvaluator,
                        'skater' : SkaterEvaluator,
                        'sklearn' : SklearnEvaluator}
        self.checks = ['idea']
        return self  
    
        
@dataclass
class Review(SimplePlan):
    """Stores machine learning experiment results.

    Review creates and stores a results report and other general
    scorers/metrics for machine learning based upon the type of model used in
    the siMpLify package. Users can manually add metrics not already included
    in the metrics dictionary by passing them to Results.edit_metric.

    Attributes:
        name: a string designating the name of the class which should be
            identical to the section of the idea with relevant settings.
        auto_finalize: sets whether to automatically call the finalize method
            when the class is instanced. If you do not draft to make any
            adjustments to the options or metrics beyond the idea, this option
            should be set to True. If you draft to make such changes, finalize
            should be called when those changes are complete.
    """

    steps : object = None
    number : int = 0
    name : str = 'review'
    auto_finalize: bool = True

    def __post_init__(self):
        self.idea_sections = ['analysis']
        super().__post_init__()
        return self

    def _set_columns(self):
        """Sets columns and options for report."""
        self.columns = {'recipe_number' : 'number',
                        'options' : 'techniques',
                        'seed' : 'seed',
                        'validation_set' : 'val_set'}
        for step in self.recipe.techniques:
            self.columns.update({step : step})
        self.columns_list = list(self.columns.keys())
        self.columns_list.extend(self.listify(self.metrics))
        self.report = pd.DataFrame(columns = self.columns_list)
        return self
    
    
    def _check_technique_name(self, step):
        """Returns appropriate algorithm to the report attribute."""
        if step.technique in ['none', 'all']:
            return step.technique
        else:
            return step.algorithm

    def _classifier_report(self):
        self.classifier_report_default = metrics.classification_report(
                self.recipe.ingredients.y_test,
                self.predictions)
        self.classifier_report_dict = metrics.classification_report(
                self.recipe.ingredients.y_test,
                self.predictions,
                output_dict = True)
        self.classifier_report = pd.DataFrame(
                self.classifier_report_dict).transpose()
        return self

    def _confusion_matrix(self):
        self.confusion = metrics.confusion_matrix(
                self.recipe.ingredients.y_test, self.predictions)
        return self
    
    def _cluster_report(self):
        return self


    def _regressor_report(self):
        return self
    
    def _format_step(self, attribute):
        if getattr(self.recipe, attribute).technique in ['none', 'all']:
            step_column = getattr(self.recipe, attribute).technique
        else:
            technique = getattr(self.recipe, attribute).technique
            parameters = getattr(self.recipe, attribute).parameters
            step_column = f'{technique}, parameters = {parameters}'
        return step_column


    def _print_classifier_results(self, recipe):
        """Prints to console basic results separate from report."""
        print('These are the results using the', recipe.model.technique,
              'model')
        if recipe.splicer.technique != 'none':
            print('Testing', recipe.splicer.technique, 'predictors')
        print('Confusion Matrix:')
        print(self.confusion)
        print('Classification Report:')
        print(self.classification_report)
        return self
    
    
    def draft(self):
        self.data_variable = 'recipes'
        return self
    
    def produce(self, recipes):
        setattr(self, self.data_variable, self.listify(recipes))
        for recipe in getattr(self, self.data_variable):
            self._check_best(recipe = recipe)
            for step, technique in self.techniques.items():
                technique.produce(recipe = recipe)
        return self