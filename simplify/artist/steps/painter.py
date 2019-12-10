"""
.. module:: painter
:synopsis: visualizations for data analysis
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from math import ceil, sqrt
import matplotlib.pyplot as plt
import pandas as pd

from simplify.creator.typesetter import SimpleDirector
from simplify.creator.typesetter import Outline


@dataclass
class Painter(SimpleDirector):
    """Creates data analysis visualizations.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    name: str = 'painter'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _check_length(self, df: pd.DataFrame, max_display: int) -> int:
        """Checks if max_display length is larger than number of columns.

        If so, number of columns in df is used instead.
        """
        if max_display > len(df.columns):
            max_display = len(df.columns)
        return max_display

    def _draft_options(self) -> None:
        self._options = CodexOptions(options = {
            'calibration': Outline(
                name = 'calibration',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {'y_true': 'y'},
                critic_dependent = {'y_pred': 'predictions.outcomes'},
                export_file = 'calibration.png'),
            'cluster_tree': Outline(
                name = 'cluster_tree',
                module = 'seaborn',
                algorithm = 'clustermap',
                data_dependent = {'data': 'x'},
                export_file = 'cluster_tree.png'),
            'confusion': Outline(
                name = 'confusion',
                module = 'seaborn',
                algorithm = 'heatmap',
                default = {'annot': True, 'fmt': 'g'},
                critic_dependent = {'data': 'reports.confusion'},
                export_file = 'confusion.png'),
            'cumulative_gain': Outline(
                name = 'cumulative_gain',
                module = 'skplt.metrics',
                algorithm = 'plot_cumulative_gain',
                data_dependent = {'y_true': 'y'},
                critic_dependent = {'y_probas': 'probabilities.outcomes'},
                export_file = 'cumulative_gain.png'),
            'decision_boundaries': Outline(
                name = 'decision_boundaries',
                module = 'mlxtend.plotting',
                algorithm = 'plot_decision_regions',
                default = {'legend': 2},
                data_dependent = {'X': 'x', 'y': 'y'},
                critic_depdent = {'clf', 'estimator'},
                export_file = 'decision_boundaries.png'),
            'elbow': Outline(
                name = 'elbow_curve',
                module = 'skplt.metrics',
                algorithm = 'plot_elbow_curve',
                data_dependent = {'X': 'x'},
                critic_depdent = {'clf', 'estimator'},
                export_file = 'elbow_curve.png'),
            'heat_map': Outline(
                name = 'heat_map',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'histogram': Outline(
                name = 'histogram',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'kde': Outline(
                name = 'kde_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'ks_statistic': Outline(
                name = 'ks_stat',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'lift': Outline(
                name = 'lift_curve',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'linear': Outline(
                name = 'linear_regress',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'logistic': Outline(
                name = 'logistic_regress',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'pair_plot': Outline(
                name = 'pair_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'pr_curve': Outline(
                name = 'pr_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'residuals': Outline(
                name = 'residuals',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'roc_curve': Outline(
                name = 'roc_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_dependency': Outline(
                name = 'shap_dependency',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_force': Outline(
                name = 'shap_force_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_heat_map': Outline(
                name = 'shap_heat_map',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_interactions': Outline(
                name = 'shap_interactions',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_summary': Outline(
                name = 'shap_summary',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'silhouette': Outline(
                name = 'silhouette',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png')}
        return self

    def heat_map(self, file_name = 'heat_map.png', **kwargs):
        seaborn.heatmap(self.x, annot = True, fmt = '.3%', **kwargs)
        self.save(file_name)
        return self

    def histogram(self, features = None, file_name = 'histogram.png',
                  **kwargs):
        for feature in listify(features):
            seaborn.distplot(self.x[feature], feature, **kwargs)
            self.save(feature + '_' + file_name)
        return self

    def kde_plot(self, file_name = 'kde_plot.png', **kwargs):
        seaborn.kdeplot(self.x, shade = True, **kwargs)
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
        seaborn.regplot(self.x, self.y, **kwargs)
        self.save(file_name)
        return self

    def logistic_regress(self, file_name = 'logit.png', **kwargs):
        seaborn.regplot(self.x, self.y, logistic = True, n_boot = 500,
                    y_jitter = .03, **kwargs)
        self.save(file_name)
        return self

    def pair_plot(self, features = None, file_name = 'pair_plot.png',
                  **kwargs):
        seaborn.pairplot(self.x, vars = features, **kwargs)
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
        seaborn.residplot(self.x, self.y, **kwargs)
        self.save(file_name)
        return self

    def roc_plot(self, file_name = 'roc_curve.png'):
        skplt.metrics.plot_roc(self.y, self.review.predicted_probs)
        self.save(file_name)
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
