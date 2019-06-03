
from dataclasses import dataclass

import pandas as pd
import sklearn.metrics as met
from eli5 import explain_prediction_df, explain_weights_df, show_prediction
from eli5 import show_weights
#import lime
from shap import DeepExplainer, KernelExplainer, LinearExplainer, TreeExplainer

from .step import Step


@dataclass
class Evaluator(Step):
    """Computes machine learning experiment scores and metrics."""
    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self,
                               sections = ['evaluator_params'])
        self.explainer_options = {'shap' : self._shap_explainer,
                                  'eli5' : self._eli5_explainer,
                                  'lime' : self._lime_explainer}
        self.shap_models = {'catboost' : 'tree',
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
                            'xgb' : 'tree'}
        self.shap_options = {'deep' : DeepExplainer,
                             'kernel' : KernelExplainer,
                             'linear' : LinearExplainer,
                             'tree' : TreeExplainer}
        return self

    @staticmethod
    def _check_none(step):
        """
        Checks if metric listed is either 'none' or 'all.' Otherwise, it
        returns the name of the algorithm selected.
        """
        if step.name in ['none', 'all']:
            return step.name
        elif not step.name:
            return 'none'
        else:
            return step.algorithm

    def _make_predictions(self, recipe):
        self.predictions = recipe.model.algorithm.predict(recipe.data.x_test)
        self.predicted_probs = recipe.model.algorithm.predict_proba(
                recipe.data.x_test)
        return self

    def _get_result(self, recipe, use_val_set):
        """
        Prepares the results of a single recipe application to be added to
        the results.table dataframe.
        """
        self.result = pd.Series(index = self.columns)
        recipe_cols = {'recipe_number' : recipe.number,
                       'step_order' : recipe.order,
                       'splicer' : recipe.splicer,
                       'scaler' : recipe.scaler,
                       'splitter' : recipe.splitter,
                       'encoder' : recipe.encoder,
                       'interactor' : recipe.interactor,
                       'sampler' : recipe.sampler,
                       'custom' : recipe.custom,
                       'selector' : recipe.selector,
                       'model' : recipe.model,
                       'seed' : recipe.model.seed,
                       'validation_set' : use_val_set}
        for key, value in recipe_cols.items():
            if key in ['recipe_number', 'step_order', 'validation_set',
                       'seed']:
                self.result[key] = value
            else:
                name = value.name
                params = value.params
                if hasattr(value, 'columns'):
                    cols = value.columns
                else:
                    cols = ['all']
                if name in ['none']:
                    self.result[key] = name
                else:
                    self.result[key] = (
                        f'{name}, parameters = {params}, columns = {cols}')
        for key, value in self.options.items():
            if key in self.metrics:
                if key in self.prob_options:
                    params = {'y_true' : recipe.data.y_test,
                              'y_prob' : self.predicted_probs[:, 1]}
                elif key in self.score_options:
                    params = {'y_true' : recipe.data.y_test,
                              'y_score' : self.predicted_probs[:, 1]}
                else:
                    params = {'y_true' : recipe.data.y_test,
                              'y_pred' : self.predictions}
                if key in self.spec_metrics:
                    params.update({key : self.spec_metrics[key]})
                result = value(**params)
                if key in self.neg_metrics:
                    result = -1 * result
                self.result[key] = result

    def _confusion(self, recipe):
        self.confusion = met.confusion_matrix(recipe.data.y_test,
                                              self.predictions)
        return self

    def _class_report(self, recipe):
        self.class_report = met.classification_report(recipe.data.y_test,
                                                      self.predictions)
        self.class_report_dict = met.classification_report(recipe.data.y_test,
                                                           self.predictions,
                                                           output_dict = True)

        self.class_report_df = pd.DataFrame(self.class_report_dict).transpose()
        return self

    def _feature_summaries(self, recipe):
        self.feature_list = list(recipe.data.x_test.columns)
        if 'svm_' in recipe.model.name or 'baseline_' in recipe.model.name:
            self.feature_import = None
        else:
            self.feature_import = pd.Series(
                    data = recipe.model.algorithm.feature_importances_,
                    index = self.feature_list)
            self.feature_import.sort_values(ascending = False,
                                            inplace = True)
        return self

    def _explain(self, recipe):
        for explainer in self._listify(self.explainers):
            explain_package = self.explainer_options[explainer]
            explain_package(recipe)
        return self

    def _shap_explainer(self, recipe):
        if recipe.model.name in self.shap_models:
            self.shap_method_type = self.shap_models[recipe.model.name]
            self.shap_method = self.shap_options[self.shap_method_type]
        elif 'baseline_' in recipe.model.name:
            self.shap_method_type = 'none'
        else:
            self.shap_method_type = 'kernel'
            self.shap_method = KernelExplainer
        data_to_explain = {'train' : recipe.data.x_train,
                           'test' : recipe.data.x_test,
                           'full' : recipe.data.x}
        df = data_to_explain[self.data_to_explain]
        if self.shap_method_type != 'none':
#            recipe.model.algorithm.fit(recipe.data.x_train,
#                                       recipe.data.y_train)
            self.shap_explainer = self.shap_method(
                    model = recipe.model.algorithm,
                    data = recipe.data.x_train)
            self.shap_values = self.shap_explainer.shap_values(df)
            if self.shap_method_type == 'tree':
                self.shap_interactions = (
                        self.shap_explainer.shap_interaction_values(
                                pd.DataFrame(df, columns = df.columns)))
            else:
                self.shap_interactions = None
        return self

    def _eli5_explainer(self, recipe):

        return self

    def _lime_explainer(self, recipe):

        return self

    def _print_results(self, recipe):
        print('These are the results using the', recipe.model.name,
              'model')
        if recipe.splicer.name != 'none':
            print('Testing', recipe.splicer.name, 'predictors')
        print('Confusion Matrix:')
        print(self.confusion)
        print('Classification Report:')
        print(self.class_report)
        return self

    def mix(self, recipe, data_to_use = 'train_test'):
        if self.verbose:
            print('Evaluating recipe')
        if data_to_use in ['train_val']:
            use_val_set = True
        else:
            use_val_set = False
        self._make_predictions(recipe)
        self._get_result(recipe, use_val_set)
        self._confusion(recipe)
        self._class_report(recipe)
        self._feature_summaries(recipe)
        self._explain(recipe)
        if self.verbose:
            self._print_results(recipe)
        return self

    def save_classification_report(self, export_path):
        self.filer.save(df = self.class_report_df,
                        export_path = export_path,
                        header = True,
                        index = True)
        return self