"""
artist.components:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import matplotlib.pyplot as plt
import seaborn as sns

import simplify
from . import base


@dataclasses.dataclass
class Styler(SimpleIterable):
    """Sets data visualization style.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    name: str = 'styler'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        """Sets fonts, colors, and styles for plots that do not have set styles.
        """
        # List of colorblind colors obtained from here:
        # https://www.dataquest.io/blog/making-538-plots/.
        # Thanks to Alex Olteanu.
        colorblind_colors = [[0,0,0], [230/255,159/255,0],
                             [86/255,180/255,233/255], [0,158/255,115/255],
                             [213/255,94/255,0], [0,114/255,178/255]]
        sns.set_style(style = self.seaborn_style)
        sns.set_context(context = self.seaborn_context)
        plt.style.use(style = self.plot_style)
        plt.rcParams['font.family'] = self.plot_font
        if self.seaborn_palette == 'colorblind':
            sns.set_palette(color_codes = colorblind_colors)
        else:
            sns.set_palette(palette = self.seaborn_palette)
        return self
    
@dataclasses.dataclass
class Painter(SimpleDirector):
    """Creates data analysis visualizations.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
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
        self._options = SimpleRepository(contents = {
            'calibration': Option(
                name = 'calibration',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {'y_true': 'y'},
                critic_dependent = {'y_pred': 'predictions.outcomes'},
                export_file = 'calibration.png'),
            'cluster_tree': Option(
                name = 'cluster_tree',
                module = 'seaborn',
                algorithm = 'clustermap',
                data_dependent = {'data': 'x'},
                export_file = 'cluster_tree.png'),
            'confusion': Option(
                name = 'confusion',
                module = 'seaborn',
                algorithm = 'heatmap',
                default = {'annot': True, 'fmt': 'g'},
                critic_dependent = {'data': 'reports.confusion'},
                export_file = 'confusion.png'),
            'cumulative_gain': Option(
                name = 'cumulative_gain',
                module = 'skplt.metrics',
                algorithm = 'plot_cumulative_gain',
                data_dependent = {'y_true': 'y'},
                critic_dependent = {'y_probas': 'probabilities.outcomes'},
                export_file = 'cumulative_gain.png'),
            'decision_boundaries': Option(
                name = 'decision_boundaries',
                module = 'mlxtend.plotting',
                algorithm = 'plot_decision_regions',
                default = {'legend': 2},
                data_dependent = {'X': 'x', 'y': 'y'},
                critic_depdent = {'clf', 'estimator'},
                export_file = 'decision_boundaries.png'),
            'elbow': Option(
                name = 'elbow_curve',
                module = 'skplt.metrics',
                algorithm = 'plot_elbow_curve',
                data_dependent = {'X': 'x'},
                critic_depdent = {'clf', 'estimator'},
                export_file = 'elbow_curve.png'),
            'heat_map': Option(
                name = 'heat_map',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'histogram': Option(
                name = 'histogram',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'kde': Option(
                name = 'kde_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'ks_statistic': Option(
                name = 'ks_stat',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'lift': Option(
                name = 'lift_curve',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'linear': Option(
                name = 'linear_regress',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'logistic': Option(
                name = 'logistic_regress',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'pair_plot': Option(
                name = 'pair_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'pr_curve': Option(
                name = 'pr_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'residuals': Option(
                name = 'residuals',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'roc_curve': Option(
                name = 'roc_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_dependency': Option(
                name = 'shap_dependency',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_force': Option(
                name = 'shap_force_plot',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_heat_map': Option(
                name = 'shap_heat_map',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_interactions': Option(
                name = 'shap_interactions',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'shap_summary': Option(
                name = 'shap_summary',
                module = 'skplt.metrics',
                algorithm = 'plot_calibration_curve',
                data_dependent = {},
                export_file = 'calibration.png'),
            'silhouette': Option(
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
        for feature in utilities.listify(features):
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
