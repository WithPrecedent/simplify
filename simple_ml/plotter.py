"""
Class for visualizing data analysis based upon the nature of the machine
learning model used in the siMpLify package.
"""
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import precision_recall_curve, roc_curve

from shap import dependence_plot, force_plot, summary_plot

from step import Step

@dataclass
class Plotter(Step):

    name : str = 'none'
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self, sections = ['plotter_params'])
        seaborn.set_style(style = self.seaborn_style)
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
        self.options = {'dependency' : self.dependency,
                        'force_plot' : self.force_plot,
                        'heat_map' : self.heat_map,
                        'histogram' : self.histogram,
                        'interactions' : self.interactions,
                        'summary' : self.summary}
        if self.model_type in ['classifier']:
            self.options.update({'confusion' : self.confusion,
                                 'pr_curve' : self.pr_plot,
                                 'roc_curve' : self.roc_plot})
            self.default_plots = ['confusion', 'heat_map', 'interactions',
                                  'pr_curve', 'roc_curve', 'summary']
        elif self.model_type in ['regressor']:
            self.default_plots = ['heat_map', 'interactions', 'summary']
        elif self.model_type in ['clusterer']:
            self.default_plots = ['heat_map', 'interactions', 'summary']
        return self


#    def _add_dependency_plots(self):
#        if self.dependency_plots in ['splices']:
#
#        return self

    def confusion(self, file_name = 'confusion_matrix.png'):

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

    def force_plot(self, file_name = 'force_plot.png'):
        force_plot(self.explainer.expected_value, self.shap_values, self.x,
                   show = False, matplotlib = True)
        return self

    def heat_map(self, file_name = 'heat_map.png', max_display = 20):
        max_display = self._check_length(self.x, max_display)
        tmp = np.abs(self.evaluator.shap_interactions).sum(0)
        for i in range(tmp.shape[0]):
            tmp[i, i] = 0
        inds = np.argsort(-tmp.sum(0))[:max_display]
        tmp2 = tmp[inds,:][:,inds]
        plt.figure(figsize = (12, 12))
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

    def histogram(self, file_name = 'histogram.png'):

        return self

    def interactions(self, file_name = 'interactions.png',  max_display = 0):
        if max_display == 0:
            max_display = self.interactions_display
        max_display = self._check_length(self.x, max_display)
        summary_plot(self.evaluator.shap_interactions, self.x,
                     max_display = max_display, show = False)
        self.save(file_name)
        return self

    def pr_plot(self, file_name = 'pr_curve.png'):

        return self

    def roc_plot(self, file_name = 'roc_curve.png'):

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