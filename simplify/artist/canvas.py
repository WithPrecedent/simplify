"""
canvas.py is the primary control file for data visualization in the siMpLify
package.

Contents:

    Canvas: class which handles construction and utilization of visuals
        to create charts, graphs, and other visuals in the siMpLify package.
    Illustration: class which stores a particular set of techniques and 
        algorithms used to create data visualizations.
        
    Both classes are subclasses to SimpleClass and follow its structural rules.
    
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

from simplify.core.base import SimpleClass


@dataclass
class Canvas(SimpleClass):
    """Visualizes data and analysis based upon the nature of the machine
    learning model used in the siMpLify package.
    """   
    ingredients : object = None
    steps : object = None
    recipes : object = None
    name : str = 'canvas'
    planner_type : str = 'serial'
    auto_finalize : bool = True
    auto_produce : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def _check_model_type(self):
        """Sets default paintings, animations, and any other added options if
        user selects 'default' as the option for the respective option.
        """
        for key in self.options.keys():
            if not hasattr(key) and getattr(self, key) == 'default':
                setattr(self, key, getattr(
                    self, '_default_' + self.model_type + 'key')())       
        return self
    
    def _default_classifier_animations(self):
        """Returns list of default animations for classifier algorithms."""
        return []
    
    def _default_classifier_paintings(self):
        """Returns list of default plots for classifier algorithms."""
        return ['confusion', 'heat_map', 'ks_statistic', 'pr_curve', 
                'roc_curve']

    def _default_cluster_animations(self):
        """Returns list of default animations for cluster algorithms."""
        return []

    def _default_cluster_paintings(self):
        """Returns list of default plots for cluster algorithms."""
        return ['cluster_tree', 'elbow', 'silhouette']

    def _default_regressor_animations(self):
        """Returns list of default animations for regressor algorithms."""
        return []

    def _default_regressor_paintings(self):
        """Returns list of default plots for regressor algorithms."""
        return ['heat_map', 'linear', 'residuals']

    def _get_ingredients(self, recipe = None):
        """
        """
        if recipe:
            return recipe.ingredients
        elif self.ingredients is not None:
            return self.ingredients
        else:
            error = 'produce method requires Ingredients or Recipe instance'
            raise TypeError(error)
    
    def _set_style(self):
        """Sets fonts, colors, and styles for plots that do not have set
        styles.
        """
        # List of colorblind colors obtained from here:
        # https://www.dataquest.io/blog/making-538-plots/.
        # Thanks to Alex Olteanu.
        colorblind_colors = [[0,0,0], [230/255,159/255,0],
                             [86/255,180/255,233/255], [0,158/255,115/255],
                             [213/255,94/255,0], [0,114/255,178/255]]
        plt.style.use(style = self.plt_style)
        plt.rcParams['font.family'] = self.plt_font
        sns.set_style(style = self.seaborn_style)
        sns.set_context(context = self.seaborn_context)
        if self.seaborn_palette == 'colorblind':
            sns.set_palette(color_codes = colorblind_colors)
        else:
            sns.set_palette(palette = self.seaborn_palette)
        return self

    def draft(self):
        """Sets default styles, options, and plots."""
        self.checks = ['steps', 'model_type']
        self._set_style()
        self.options = {'painter' : Paint,
                        'animator' : Animate}
        return self

#    def _edit_dependency_plots(self):
#        if self.dependency_plots in ['cleaves']:
#
#        return self

    def finalize(self):
        self.visuals = []
        step_combinations = []
        for step in self.options.keys():
            # Stores each step attribute in a list
            setattr(self, step, self.listify(getattr(self, step)))
            # Adds step to a list of all step lists
            step_combinations.append(getattr(self, step))    
        return self
    
    def produce(self, recipes = None, reviews = None):
        if self.verbose:
            print('Creating and exporting visuals')
        for i, recipe in enumerate(recipes):
            ingredients = self._get_ingredients(recipe = recipe)
            estimator = recipe.model.algorithm
            ingredients._remap_dataframes(data_to_use = self.data_to_plot)
            x = ingredients.x_test
            y = ingredients.y_test
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
    
    
@dataclass
class Illustration(SimpleClass):
    
    def __post_init__(self):
        super().__post_init__()
        return self