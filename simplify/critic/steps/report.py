
from dataclasses import dataclass

import pandas as pd

from simplify.core.base import SimpleStep


@dataclass
class Report(SimpleStep):


    def _set_columns(self):
        """Sets columns and options for report."""
        self.columns = {'recipe_number' : 'number',
                        'options' : 'techniques',
                        'seed' : 'seed',
                        'validation_set' : 'val_set'}
        for step in self.recipe.techniques:
            self.columns.update({step : step})
        self.columns_list = list(self.columns.keys())
        self.columns_list.extend(self.listify(self.metrics))
        self.report = pd.DataFrame(columns = self.columns_list)
        return self
    
    
    def _check_technique_name(self, step):
        """Returns appropriate algorithm to the report attribute."""
        if step.technique in ['none', 'all']:
            return step.technique
        else:
            return step.algorithm

    def _format_step(self, attribute):
        if getattr(self.recipe, attribute).technique in ['none', 'all']:
            step_column = getattr(self.recipe, attribute).technique
        else:
            technique = getattr(self.recipe, attribute).technique
            parameters = getattr(self.recipe, attribute).parameters
            step_column = f'{technique}, parameters = {parameters}'
        return step_column


    def _print_classifier_results(self, recipe):
        """Prints to console basic results separate from report."""
        print('These are the results using the', recipe.model.technique,
              'model')
        if recipe.splicer.technique != 'none':
            print('Testing', recipe.splicer.technique, 'predictors')
        print('Confusion Matrix:')
        print(self.confusion)
        print('Classification Report:')
        print(self.classification_report)
        return self