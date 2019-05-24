"""
Class for visualizing data analysis based upon the nature of the machine
learning model used in the siMpLify package.
"""
from dataclasses import dataclass
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
        self._check_plots()
        return self

    def _check_plots(self):
        if self.plotter in ['default']:
            self.plots = self.default_plots
        return self

    def _check_length(self, df, max_display):
        if max_display > len(df.columns):
            max_display = len(df.columns)
        return max_display

    def _set_options(self):
        self.options = {'calibration' : self.calibration,
                        'cluster_tree' : self.cluster_tree,
                        'confusion' : self.confusion,
                        'cumulative_gain' : self.cumulative,
                        'dependency' : self.dependency,
                        'elbow' : self.elbow_curve,
                        'force_plot' : self.force_plot,
                        'heat_map' : self.heat_map,
                        'histogram' : self.histogram,
                        'interactions' : self.interactions,
                        'kde' : self.kde_plot,
                        'ks_statistic' : self.ks_stat,
                        'lift' : self.lift_curve,
                        'linear' : self.linear_regress,
                        'logistic' : self.logistic_regress,
                        'pair_plot' : self.pair_plot,
                        'pr_curve' : self.pr_plot,
                        'residuals' : self.residuals,
                        'roc_curve' : self.roc_plot,
                        'silhouette' : self.silhouette,
                        'summary' : self.summary}
        if self.model_type in ['classifier']:
            self.default_plots = ['confusion', 'heat_map',
                                  'interactions', 'ks_statistic', 'pr_curve',
                                  'roc_curve', 'summary']
        elif self.model_type in ['regressor']:
            self.default_plots = ['heat_map', 'interactions', 'linear',
                                  'residuals', 'summary']
        elif self.model_type in ['clusterer']:
            self.default_plots = ['cluster_tree', 'elbow', 'silhouette']
        return self


#    def _add_dependency_plots(self):
#        if self.dependency_plots in ['splices']:
#
#        return self

    def calibration(self, file_name = 'calibration.png', probs_list = None,
                    model_list = None):
        if model_list:
            skplt.metrics.plot_calibration_curve(
                    self.y, probs_list, model_list)
        else:
            skplt.metrics.plot_calibration_curve(
                    self.y, self.recipe.evaluator.predicted_probs)
        self.save(file_name)
        return self

    def cluster_tree(self, file_name = 'cluster_tree.png', **kwargs):
        sns.clustermap(self.x, **kwargs)
        self.save(file_name)
        return self

    def confusion(self, file_name = 'confusion_matrix.png'):

        return self

    def cumulative(self, file_name = 'cumulative_gain.png'):
        skplt.metrics.plot_cumulative_gain(
                self.y, self.recipe.evaluator.predicted_probs)
        self.save(file_name)
        return self

    def dependency(self, model, var1, var2 = None, x = None,
                   file_name = 'shap_dependency.png'):
        if var2:
            dependence_plot(var1, self.shap_values, x,
                            interaction_index = 'var2', show = False,
                            matplotlib = True)
        else:
            dependence_plot(var1, self.shap_values, x, show = False,
                            matplotlib = True)
        self.save(file_name)
        return self

    def elbow_curve(self, file_name = 'elbow_curve.png'):
        skplt.metrics.plot_elbow_curve(self.recipe.model.algorithm, self.x)
        self.save(file_name)
        return self

    def force_plot(self, file_name = 'force_plot.png'):
        force_plot(self.evaluator.explainer.expected_value, self.shap_values,
                   self.x, show = False, matplotlib = True)
        self.save(file_name)
        return self

    def histogram(self, feature, file_name = 'histogram.png', **kwargs):
        sns.distplot(self.x[feature], feature, **kwargs)
        return self

    def heat_map(self, file_name = 'heat_map.png', **kwargs):
        sns.heatmap(self.x, **kwargs)
        self.save(file_name)
        return self

    def interactions(self, file_name = 'interactions.png',  max_display = 0):
        if max_display == 0:
            max_display = self.interactions_display
        max_display = self._check_length(self.x, max_display)
        summary_plot(self.evaluator.shap_interactions, self.x,
                     max_display = max_display, show = False)
        self.save(file_name)
        return self

    def kde_plot(self, file_name = 'kde_plot.png', **kwargs):
        sns.kdeplot(self.x, shade = True, **kwargs)
        self.save(file_name)
        return self

    def ks_stat(self, file_name = 'ks_stat.png'):
        skplt.metrics.plot_ks_statistic(
                self.y, self.recipe.evaluator.predicted_probs)
        self.save(file_name)
        return self

    def lift_curve(self, file_name = 'lift_curve.png'):
        skplt.metrics.plot_lift_curve(
                self.y, self.recipe.evaluator.predicted_probs)
        self.save(file_name)
        return self

    def linear_regress(self, file_name = 'linear_regression.png', **kwargs):
        sns.regplot(self.x, self.y, **kwargs)
        self.save(file_name)
        return self

    def logistic_regress(self, file_name = 'logit.png', **kwargs):
        sns.regplot(self.x, self.y, logistic = True, n_boot = 500,
                    y_jitter = .03, **kwargs)
        self.save(file_name)
        return self

    def pair_plot(self, features, file_name = 'pair_plot.png', **kwargs):
        sns.pairplot(self.x, vars = features, **kwargs)
        self.save(file_name)
        return self

    def pr_plot(self, file_name = 'pr_curve.png'):
        skplt.metrics.plot_precision_recall_curve(
                self.y, self.recipe.evaluator.predicted_probs)
        self.save(file_name)
        return self

    def residuals(self, file_name = 'residuals.png', **kwargs):
        sns.residplot(self.x, self.y, **kwargs)
        self.save(file_name)
        return self

    def roc_plot(self, file_name = 'roc_curve.png'):
        skplt.metrics.plot_roc(
                self.y, self.recipe.evaluator.predicted_probs)
        self.save(file_name)
        return self

    def shap_heat_map(self, file_name = 'shap_heat_map.png', max_display = 10):
        max_display = self._check_length(self.x, max_display)
        tmp = np.abs(self.evaluator.shap_interactions).sum(0)
        for i in range(tmp.shape[0]):
            tmp[i, i] = 0
        inds = np.argsort(-tmp.sum(0))[:30]
        tmp2 = tmp[inds,:][:,inds]
        plt.figure(figsize = (max_display, max_display))
        plt.imshow(tmp2)
        plt.yticks(range(tmp2.shape[0]),
                   self.x.columns[inds],
                   rotation = 50.4,
                   horizontalalignment = 'right')
        plt.xticks(range(tmp2.shape[0]),
                   self.x.columns[inds],
                   rotation = 50.4,
                   horizontalalignment = 'left')
        plt.gca().xaxis.tick_top()
        self.save(file_name)
        return self

    def silhouette(self, file_name = 'silhouette.png'):
        skplt.metrics.plot_silhouette(
                self.x, self.recipe.model.algorithm.labels_)
        self.save(file_name)
        return self

    def summary(self, file_name = 'shap_summary.png', max_display = 0):
        if max_display == 0:
            max_display = self.summary_display
        summary_plot(self.evaluator.shap_values,
                     self.x,
                     max_display = max_display,
                     show = False)
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

    def mix(self, recipe, evaluator):
        if self.verbose:
            print('Creating and exporting visuals')
        self.recipe = recipe
        self.evaluator = evaluator
        self.x, self.y = self.recipe.data[self.data_to_plot]
#        if self.dependency_plots != 'none':
#            self._add_dependency_plots()
        for plot in self.plots:
            self.options[plot]()
        return self