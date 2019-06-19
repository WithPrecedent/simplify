
from dataclasses import dataclass

import pandas as pd
import sklearn.metrics as met
from eli5 import explain_prediction_df, explain_weights_df, show_prediction
from eli5 import show_weights
#import lime
from shap import DeepExplainer, KernelExplainer, LinearExplainer, TreeExplainer

from ..cookbook.ingredient import Ingredient


@dataclass
class Evaluator(Ingredient):
    """Computes machine learning experiment scores and metrics."""

    technique : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.settings.localize(instance = self,
                               sections = ['evaluator_params'])

        return self



    def blend(self, recipe, data_to_use = 'train_test'):
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
#        self.filer.save(df = self.class_report_df,
#                        export_path = export_path,
#                        header = True,
#                        index = True)
        return self