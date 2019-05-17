"""
Class for visualizing data analysis based upon the nature of the machine
learning model used.
"""


from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap import dependence_plot, force_plot, summary_plot
from shap import DeepExplainer, KernelExplainer, LinearExplainer, TreeExplainer

from step import Step

@dataclass
class Plotter(Step):

    name : str = 'default'
    params : object = None
    model : object = None
    export_path : str = ''

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self, sections = ['plotter_params'])
        if self.model_type in ['classifier']:
            self.options = {'heat_map' : self.heat_map,
                            'summary' : self.summary,
                            'interactions' : self.interactions}
            if self.model.name in ['xgb', 'random_forest']:
                self.explainer = TreeExplainer
            elif self.model.name in ['logit']:
                self.explainer = LinearExplainer
            elif self.model.name in ['torch', 'tensor_flow']:
                self.explainer = DeepExplainer
            else:
                self.explainer = KernelExplainer
        elif self.model_type in ['regressor']:
            self.options = {'heat_map' : self.heat_map,
                            'summary' : self.summary,
                            'interactions' : self.interactions}
            if self.model.name in ['torch', 'tensor_flow']:
                self.explainer = DeepExplainer
            else:
                self.explainer = LinearExplainer
        else:
            self.options = {'heat_map' : self.heat_map,
                            'summary' : self.summary,
                            'interactions' : self.interactions}
            self.explainer = KernelExplainer
        return self

    def _check_plots(self):
        if self.plotter in ['default']:
            self.plots = self.options.keys()
        return self

#    def _add_dependency_plots(self):
#        if self.dependency_plots in ['splices']:
#
#        return self

    def _iter_plots(self):
        self._compute_shap_values()
        self._check_plots()
#        if self.dependency_plots != 'none':
#            self._add_dependency_plots()
        for plot in self.plots:
            self.options[plot]()
        return self

    def _compute_shap_values(self):
        self.shap_values = self.explainer(
                    self.recipe.model.algorithm).shap_values(self.x)
        self.interaction_values = self.explainer(
                    self.recipe.model.algorithm).shap_interaction_values(
                            pd.DataFrame(self.x, columns = self.x.columns))
        return self

    def visualize(self, recipe):
        self.recipe = recipe
        self.x, self.y = self.recipe.data[self.data_to_use]
        self._iter_plots()
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

    def heat_map(self, file_name = 'heat_map.png'):
        tmp = np.abs(self.interaction_values).sum(0)
        for i in range(tmp.shape[0]):
            tmp[i, i] = 0
        inds = np.argsort(-tmp.sum(0))[:20]
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

    def interactions(self, file_name = 'interactions.png',  max_display = 0):
        if max_display == 0:
            max_display = self.interactions_display
        summary_plot(self.interaction_values, self.x,
                     max_display = max_display, show = False)
        self.save(file_name)
        return self

    def summary(self, file_name = 'shap_summary.png', max_display = 0):
        if max_display == 0:
            max_display = self.summary_display
        summary_plot(self.shap_values, self.x, max_display = max_display,
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