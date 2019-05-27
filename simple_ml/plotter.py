"""
Class for visualizing data analysis based upon the nature of the machine
learning model used in the siMpLify package.
"""
from dataclasses import dataclass
from functools import wraps
from inspect import getfullargspec, signature
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from shap import dependence_plot, force_plot, summary_plot
import scikitplot as skplt

from step import Step

@dataclass
class Plotter(Step):

    name : str = 'none'
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self, sections = ['plotter_params'])
        sns.set_style(style = self.seaborn_style)
        self._set_options()
        self._set_plots()
        return self

    def _set_options(self):
        self.options = {'calibration' : self.calibration,
                        'cluster_tree' : self.cluster_tree,
                        'confusion' : self.confusion,
                        'cumulative_gain' : self.cumulative,
                        'elbow' : self.elbow_curve,
                        'heat_map' : self.heat_map,
                        'histogram' : self.histogram,
                        'kde' : self.kde_plot,
                        'ks_statistic' : self.ks_stat,
                        'lift' : self.lift_curve,
                        'linear' : self.linear_regress,
                        'logistic' : self.logistic_regress,
                        'pair_plot' : self.pair_plot,
                        'pr_curve' : self.pr_plot,
                        'residuals' : self.residuals,
                        'roc_curve' : self.roc_plot,
                        'shap_dependency' : self.shap_dependency,
                        'shap_force' : self.shap_force_plot,
                        'shap_heat_map' : self.shap_heat_map,
                        'shap_interactions' : self.shap_interactions,
                        'shap_summary' : self.shap_summary,
                        'silhouette' : self.silhouette}
        if self.model_type in ['classifier']:
            self.default_plots = ['confusion', 'heat_map','ks_statistic',
                                  'pr_curve', 'roc_curve']
        elif self.model_type in ['regressor']:
            self.default_plots = ['heat_map', 'linear', 'residuals']
        elif self.model_type in ['clusterer']:
            self.default_plots = ['cluster_tree', 'elbow', 'silhouette']
        return self

    def _check_length(self, df, max_display):
        if max_display > len(df.columns):
            max_display = len(df.columns)
        return max_display

#    def _add_dependency_plots(self):
#        if self.dependency_plots in ['splices']:
#
#        return self

    def _set_plots(self):
        if self.plotter in ['default']:
            self.plots = self.default_plots
        return self

    def set_defaults(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(func)
            test_vars = ['recipe', 'x', 'y', 'predicted_probs', 'estimator']
            unpassed_args = argspec.args[len(args):]
            sig = dict(signature(func).bind(test_vars).arguments)
            if 'recipe' in unpassed_args:
                for test_var in test_vars:
                    if test_var in argspec.args and test_var in unpassed_args:
                        kwargs.update({test_var : getattr(self, test_var)})
            elif 'recipe' in argspec.args:
                kwargs.update({'recipe' : sig['recipe']})
                x, y = sig['recipe'].data[self.data_to_plot]
                if 'x' in argspec.args and 'x' in unpassed_args:
                    kwargs.update({'x' : x})
                if 'y' in argspec.args and 'y' in unpassed_args:
                    kwargs.update({'y' : y})
                if ('predicted_probs' in argspec.args
                        and 'predicted_probs' in unpassed_args):
                    new_param = getattr(sig['recipe'],
                                        'evalutor.predicted_probs')
                    kwargs.update({'predicted_probs' : new_param})
                if ('estimator' in argspec.args
                        and 'predicted_probs' in unpassed_args):
                    new_param = getattr(sig['recipe'], 'model.algorithm')
                    kwargs.update({'estimator' : new_param})
            return func(self, *args, **kwargs)
        return wrapper

    @set_defaults
    def calibration(self, recipe = None, y = None, probs_list = None,
                    model_list = None, file_name = 'calibration.png'):
        skplt.metrics.plot_calibration_curve(y, probs_list, model_list)
        self.save(file_name)
        return self

    @set_defaults
    def cluster_tree(self, recipe = None, x = None,
                     file_name = 'cluster_tree.png', **kwargs):
        sns.clustermap(x, **kwargs)
        self.save(file_name)
        return self

    @set_defaults
    def confusion(self, recipe = None, file_name = 'confusion_matrix.png'):
        sns.heatmap(recipe.evaluator.confusion, annot = True, fmt = 'g')
        self.save(file_name)
        return self

    @set_defaults
    def cumulative(self, recipe = None, y = None, predicted_probs = None,
                   file_name = 'cumulative_gain.png'):
        skplt.metrics.plot_cumulative_gain(y, predicted_probs)
        self.save(file_name)
        return self

    @set_defaults
    def elbow_curve(self, recipe = None, x = None, estimator = None,
                    file_name = 'elbow_curve.png'):
        skplt.metrics.plot_elbow_curve(estimator, x)
        self.save(file_name)
        return self

    @set_defaults
    def heat_map(self, recipe = None, x = None, file_name = 'heat_map.png',
                 **kwargs):
        sns.heatmap(x, fmt = '.3%', **kwargs)
        self.save(file_name)
        return self

    @set_defaults
    def histogram(self, recipe = None, x = None, feature = None,
                  file_name = 'histogram.png', **kwargs):
        sns.distplot(x[feature], feature, **kwargs)
        return self

    @set_defaults
    def kde_plot(self, recipe = None, x = None, file_name = 'kde_plot.png',
                 **kwargs):
        sns.kdeplot(x, shade = True, **kwargs)
        self.save(file_name)
        return self

    @set_defaults
    def ks_stat(self, recipe = None, y = None, predicted_probs = None,
                file_name = 'ks_stat.png'):
        skplt.metrics.plot_ks_statistic(y, predicted_probs)
        self.save(file_name)
        return self

    @set_defaults
    def lift_curve(self, recipe = None, y = None, predicted_probs = None,
                   file_name = 'lift_curve.png'):
        skplt.metrics.plot_lift_curve(y, predicted_probs)
        self.save(file_name)
        return self

    @set_defaults
    def linear_regress(self, recipe = None, x = None, y = None,
                       file_name = 'linear_regression.png', **kwargs):
        sns.regplot(x, y, **kwargs)
        self.save(file_name)
        return self

    @set_defaults
    def logistic_regress(self, recipe = None, x = None, y = None,
                         file_name = 'logit.png', **kwargs):
        sns.regplot(x, y, logistic = True, n_boot = 500, y_jitter = .03,
                    **kwargs)
        self.save(file_name)
        return self

    @set_defaults
    def pair_plot(self, recipe = None, x = None, features = None,
                  file_name = 'pair_plot.png', **kwargs):
        sns.pairplot(x, vars = features, **kwargs)
        self.save(file_name)
        return self

    @set_defaults
    def pr_plot(self, recipe = None, y = None, predicted_probs = None,
                file_name = 'pr_curve.png'):
        skplt.metrics.plot_precision_recall_curve(y, predicted_probs)
        self.save(file_name)
        return self

    @set_defaults
    def residuals(self, recipe = None, x = None, y = None,
                  file_name = 'residuals.png', **kwargs):
        sns.residplot(x, y, **kwargs)
        self.save(file_name)
        return self

    @set_defaults
    def roc_plot(self, recipe = None, y = None, predicted_probs = None,
                 file_name = 'roc_curve.png'):
        skplt.metrics.plot_roc(y, predicted_probs)
        self.save(file_name)
        return self

    @set_defaults
    def shap_dependency(self, recipe = None, x = None, var1 = None,
                        var2 = None, file_name = 'shap_dependency.png'):
        if var2:
            dependence_plot(var1, recipe.evaluator.shap_values, x,
                            interaction_index = 'var2', show = False,
                            matplotlib = True)
        else:
            dependence_plot(var1, recipe.evaluator.shap_values, x,
                            show = False, matplotlib = True)
        self.save(file_name)
        return self

    @set_defaults
    def shap_force_plot(self, recipe = None, x = None,
                        file_name = 'shap_force_plot.png'):
        force_plot(recipe.evaluator.explainer.expected_value,
                   recipe.evaluator.shap_values, x, show = False,
                   matplotlib = True)
        self.save(file_name)
        return self

    @set_defaults
    def shap_heat_map(self, recipe = None, x = None,
                      file_name = 'shap_heat_map.png', max_display = 10):
        max_display = self._check_length(x, max_display)
        tmp = np.abs(recipe.evaluator.shap_interactions).sum(0)
        for i in range(tmp.shape[0]):
            tmp[i, i] = 0
        inds = np.argsort(-tmp.sum(0))[:30]
        tmp2 = tmp[inds,:][:,inds]
        plt.figure(figsize = (max_display, max_display))
        plt.imshow(tmp2)
        plt.yticks(range(tmp2.shape[0]),
                   x.columns[inds],
                   rotation = 50.4,
                   horizontalalignment = 'right')
        plt.xticks(range(tmp2.shape[0]),
                   x.columns[inds],
                   rotation = 50.4,
                   horizontalalignment = 'left')
        plt.gca().xaxis.tick_top()
        self.save(file_name)
        return self

    @set_defaults
    def shap_interactions(self, recipe = None, x = None,
                          file_name = 'shap_interactions.png',
                          max_display = 0):
        if max_display == 0:
            max_display = self.interactions_display
        max_display = self._check_length(x, max_display)
        summary_plot(recipe.evaluator.shap_interactions, x,
                     max_display = max_display, show = False)
        self.save(file_name)
        return self

    @set_defaults
    def shap_summary(self, recipe = None, x = None,
                     file_name = 'shap_summary.png', max_display = 0):
        if max_display == 0:
            max_display = self.summary_display
        summary_plot(recipe.evaluator.shap_values, x,
                     max_display = max_display, show = False)
        self.save(file_name)
        return self

    @set_defaults
    def silhouette(self, recipe = None, estimator = None, x = None,
                   file_name = 'silhouette.png'):
        skplt.metrics.plot_silhouette(x, estimator.labels_)
        self.save(file_name)
        return self

    def save(self, file_name):
        export_path = self.filer._iter_path(model = self.recipe.model,
                                            recipe_number = self.recipe.number,
                                            splicer = self.recipe.splicer,
                                            file_name = file_name,
                                            file_type = 'png')
        plt.savefig(export_path, bbox_inches = 'tight')
        plt.close()
        return self

    def mix(self, recipe, plot_list = None):
        if self.verbose:
            print('Creating and exporting visuals')
        self.recipe = recipe
        self.estimator = self.recipe.model.algorithm
        if hasattr(self.recipe.evaluator, 'predicted_probs'):
            self.predicted_probs = self.recipe.evaluator.predicted_probs
        else:
            self.predicted_probs = None
        self.x, self.y = self.recipe.data[self.data_to_plot]
        if ('shap' in self.settings['evaluator_params']['explainers']
            and self.name == 'default'):
            self.plots.extend(['shap_heat_map', 'shap_summary'])
            if self.recipe.evaluator.shap_method_type == 'tree':
                self.plots.append('shap_interactions')
#        if self.dependency_plots != 'none':
#            self._add_dependency_plots()
        for plot in self.plots:
            self.options[plot]()
        return self