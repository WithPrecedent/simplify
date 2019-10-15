"""
.. module:: paint
:synopsis: visualizations for data analysis
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from math import ceil, sqrt
import matplotlib.pyplot as plt

from simplify.core.iterable import SimpleIterable


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'calibration': self.calibration,
    'cluster_tree': self.cluster_tree,
    'confusion': self.confusion,
    'cumulative_gain': self.cumulative,
    'decision_boundaries': self.decision_boundaries,
    'elbow': self.elbow_curve,
    'heat_map': self.heat_map,
    'histogram': self.histogram,
    'kde': self.kde_plot,
    'ks_statistic': self.ks_stat,
    'lift': self.lift_curve,
    'linear': self.linear_regress,
    'logistic': self.logistic_regress,
    'pair_plot': self.pair_plot,
    'pr_curve': self.pr_plot,
    'residuals': self.residuals,
    'roc_curve': self.roc_plot,
    'shap_dependency': self.shap_dependency,
    'shap_force': self.shap_force_plot,
    'shap_heat_map': self.shap_heat_map,
    'shap_interactions': self.shap_interactions,
    'shap_summary': self.shap_summary,
    'silhouette': self.silhouette}


@dataclass
class Paint(SimpleIterable):
    """Creates data analysis visualizations.

    Args:
        steps(dict(str: SimpleTechnique)): names and related SimpleTechnique classes for
            creating data visualizations.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.
    """

    steps: object = None
    name: str = 'paint'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def __post_init__(self):
        super().__post_init__()
        return self

    def _check_length(self, df, max_display):
        """Checks if max_display length is larger than number of columns.
        If so, number of columns in df is used instead.
        """
        if max_display > len(df.columns):
            max_display = len(df.columns)
        return max_display

    def _set_grid(self, recipes):
        size = ceil(sqrt(self.len(recipes)))
        self.grid = gridspec.GridSpec(size, size)
        return self

    def calibration(self, file_name = 'calibration.png'):
        skplt.metrics.plot_calibration_curve(self.y, self.review.probs_list,
                                             self.review.model_list)
        self.save(file_name)
        return self

    def cluster_tree(self, file_name = 'cluster_tree.png', **kwargs):
        sns.clustermap(self.x, **kwargs)
        self.save(file_name)
        return self

    def confusion(self, file_name = 'confusion_matrix.png'):
        sns.heatmap(self.review.confusion, annot = True, fmt = 'g')
        self.save(file_name)
        return self

    def cumulative(self, file_name = 'cumulative_gain.png'):
        skplt.metrics.plot_cumulative_gain(self.y, self.review.predicted_probs)
        self.save(file_name)
        return self

    def decision_boundaries(self, file_name = 'decision_boundaries.png'):
        plot_decision_regions(X = self.x, y = self.y, clf = self.estimator,
                              legend = 2)
        self.save(file_name)
        return self

    def draft(self):
        """Sets available plots dictionary."""
        return self

#    def _edit_dependency_plots(self):
#        if self.dependency_plots in ['cleaves']:
#
#        return self

    def elbow_curve(self, file_name = 'elbow_curve.png'):
        skplt.metrics.plot_elbow_curve(self.estimator, self.x)
        self.save(file_name)
        return self

    def heat_map(self, file_name = 'heat_map.png', **kwargs):
        sns.heatmap(self.x, annot = True, fmt = '.3%', **kwargs)
        self.save(file_name)
        return self

    def histogram(self, features = None, file_name = 'histogram.png',
                  **kwargs):
        for feature in self.listify(features):
            sns.distplot(self.x[feature], feature, **kwargs)
            self.save(feature + '_' + file_name)
        return self

    def kde_plot(self, file_name = 'kde_plot.png', **kwargs):
        sns.kdeplot(self.x, shade = True, **kwargs)
        self.save(file_name)
        return self

    def ks_stat(self, file_name = 'ks_stat.png'):
        skplt.metrics.plot_ks_statistic(self.y, self.review.predicted_probs)
        self.save(file_name)
        return self

    def lift_curve(self, file_name = 'lift_curve.png'):
        skplt.metrics.plot_lift_curve(self.y, self.review.predicted_probs)
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

    def pair_plot(self, features = None, file_name = 'pair_plot.png',
                  **kwargs):
        sns.pairplot(self.x, vars = features, **kwargs)
        self.save(file_name)
        return self

    def publish(self):
        return self

    def pr_plot(self, file_name = 'pr_curve.png'):
        skplt.metrics.plot_precision_recall_curve(self.y,
                                                  self.review.predicted_probs)
        self.save(file_name)
        return self

    def residuals(self, file_name = 'residuals.png', **kwargs):
        sns.residplot(self.x, self.y, **kwargs)
        self.save(file_name)
        return self

    def roc_plot(self, file_name = 'roc_curve.png'):
        skplt.metrics.plot_roc(self.y, self.review.predicted_probs)
        self.save(file_name)
        return self

    def save(self, file_name):
        self.depot.save(variable = plt,
                            folder = self.depot.recipe,
                            file_name = file_name,
                            file_format = 'png')
        return self

    def shap_dependency(self, var1 = None, var2 = None,
                        file_name = 'shap_dependency.png'):
        if self.review.shap_method_type != 'none':
            if var2:
                dependence_plot(var1, self.review.shap_values, self.x,
                                interaction_index = 'var2', show = False,
                                matplotlib = True)
            else:
                dependence_plot(var1, self.review.shap_values, self.x,
                                show = False, matplotlib = True)
            self.save(file_name)
        return self

    def shap_force_plot(self, file_name = 'shap_force_plot.png'):
        if self.review.shap_method_type != 'none':
            force_plot(self.review.explainer.expected_value,
                       self.review.shap_values, self.x, show = False,
                       matplotlib = True)
            self.save(file_name)
        return self

    def shap_heat_map(self, file_name = 'shap_heat_map.png', max_display = 10):
        if self.review.shap_method_type == 'tree':
            max_display = self._check_length(self.x, max_display)
            tmp = np.abs(self.review.shap_interactions).sum(0)
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

    def shap_interactions(self, file_name = 'shap_interactions.png',
                          max_display = 0):
        if self.review.shap_method_type == 'tree':
            if max_display == 0:
                max_display = self.interactions_display
            max_display = self._check_length(self.x, max_display)
            summary_plot(self.review.shap_interactions, self.x,
                         max_display = max_display, show = False)
            self.save(file_name)
        return self

    def shap_summary(self, file_name = 'shap_summary.png', max_display = 0):
        if self.review.shap_method_type != 'none':
            if max_display == 0:
                max_display = self.summary_display
            summary_plot(self.review.shap_values, self.x,
                         max_display = max_display, show = False)
            self.save(file_name)
        return self

    def silhouette(self, file_name = 'silhouette.png'):
        skplt.metrics.plot_silhouette(self.x, self.estimator.labels_)
        self.save(file_name)
        return self

    def implement(self, recipes, reviews):
        if ('shap' in self.explainers
            and self.presentation_options == 'default'):
            self.plots.extend(['shap_heat_map', 'shap_summary'])
            if self.review.shap_method_type == 'tree':
                self.plots.append('shap_interactions')
#        if self.dependency_plots != 'none':
#            self._edit_dependency_plots()
        for plot in self.plots:
            self.options[plot]()
        return self
