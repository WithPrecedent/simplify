
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.base import SimplePlan, SimpleStep


@dataclass
class Explain(SimplePlan):
    """Core class for evaluating the results of data analysis produceed by
    the siMpLify Cookbook.

    Args:

    """
    steps : object = None
    name : str = 'explainer'
    auto_finalize : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _get_importances(self, step_instance, recipe):
        return step_instance._produce_importances(recipe = recipe)

    def _get_reports(self, step_instance, recipe):
        return step_instance._produce_reports(recipe = recipe)

    """ Core siMpLify Methods """

    def draft(self):
        self.options = {
                'eli5': Eli5Explain,
                'shap': ShapExplain,
                'skater': SkaterExplain}
        self.checks = ['steps']
        self.custom_options = list(self.options.keys())
        self.importances_options = list(self.options.keys())
        self.reports_options = ['eli5', 'shap']
        return self

    def produce(self, recipe):
        """Creates a dictionary of 'reports' from explainer techniques.

        Args:
            recipe (Recipe): a Recipe with a fitted model.
        """
        self.importances = {}
        self.reports = {}
        for step_name, step_instance in self.options.items():
            if step_name in self.steps:
                for return_value in ('importances', 'reports'):
                    if step_name in getattr(self, return_value + '_options'):
                        produced = getattr(self, '_get_' + return_value)(
                                step_instance = step_instance,
                                recipe = recipe)
                        getattr(self, return_value).update({
                                step_name : produced})
        return self

@dataclass
class Eli5Explain(SimpleStep):
    """Explains fit model with eli5 package.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique : object = None
    parameters : object = None
    name : str = 'eli5'
    auto_finalize : bool = True

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _produce_specific(self, recipe):
        return self


    """ Core siMpLify Methods """

    def draft(self):
        # Local import to save memory if not used.
        from eli5 import explain_prediction_df
        from eli5.sklearn import PermutationImportance

        super().draft()
        self.options = {'feature' : explain_prediction_df,
                        'permutation' : PermutationImportance}
        self.models = {'baseline' : 'none',
                       'catboost' : 'specific',
                       'decision_tree' : 'specific',
                       'lasso' : 'specific',
                       'lasso_lars' : 'specific',
                       'light_gbm' : 'specific',
                       'logit' : 'specific',
                       'ols' : 'specific',
                       'random_forest' : 'specific',
                       'ridge' : 'specific',
                       'svm_linear' : 'specific',
                       'tensor_flow' : 'permutation',
                       'torch' : 'permutation',
                       'xgboost' : 'specific'}
        return self

    def produce(self):

        self.permutation_weights = show_weights(
                self.permutation_importances,
                feature_names = recipe.ingredients.columns.keys())
        return self

@dataclass
class ShapExplain(SimpleStep):
    """Explains fit model with shap package.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique : object = None
    parameters : object = None
    name : str = 'shap'
    auto_finalize : bool = True

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        # Local import to save memory if not used.
        from shap import (DeepExplainer, KernelExplainer, LinearExplainer,
                          TreeExplainer)

        self.options = {'deep' : DeepExplainer,
                        'kernel' : KernelExplainer,
                        'linear' : LinearExplainer,
                        'tree' : TreeExplainer}
        self.models = {'baseline' : 'none',
                       'catboost' : 'tree',
                       'decision_tree' : 'tree',
                       'lasso' : 'linear',
                       'lasso_lars' : 'linear',
                       'light_gbm' : 'tree',
                       'logit' : 'linear',
                       'ols' : 'linear',
                       'random_forest' : 'tree',
                       'ridge' : 'linear',
                       'svm_linear' : 'linear',
                       'tensor_flow' : 'deep',
                       'torch' : 'deep',
                       'xgboost' : 'tree'}
        return self

    def produce(self, recipe):
        """Applies shap evaluator to data based upon type of model used."""
        if recipe.model.technique in self.shap_models:
            self.shap_method_type = self.shap_models[
                    recipe.model.technique]
            self.shap_method = self.shap_options[self.shap_method_type]
        else:
            self.shap_method_type = 'kernel'
        df = self.options[self.data_to_evaluate]
        if self.shap_method_type != 'none':
            self.shap_evaluator = self.shap_method(
                    model = recipe.model.algorithm,
                    data = recipe.ingredients.x_train)
            self.shap_values = self.shap_evaluator.shap_values(df)
            if self.shap_method_type == 'tree':
                self.shap_interactions = (
                        self.shap_evaluator.shap_interaction_values(
                                pd.DataFrame(df, columns = df.columns)))
            else:
                self.shap_interactions = None
        self.feature_importances = np.abs(self.shap_values).mean(0)
        return self


@dataclass
class SkaterExplain(SimpleStep):
    """Explains fit model with skater package.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique : object = None
    parameters : object = None
    name : str = 'skater'
    auto_finalize : bool = True

    def __post_init__(self):
        """Sets up the core attributes of a ShapEvaluator instance."""
        super().__post_init__()
        return self

    def draft(self):
        # Local import to save memory if not used.
        return self
