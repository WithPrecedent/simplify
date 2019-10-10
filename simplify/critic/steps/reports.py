
"""
.. module:: reports
:synopsis: reports for model performance
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleTechnique


@dataclass
class Reports(SimpleTechnique):

    recipe : object = None
    technique: object = None
    parameters: object = None
    name: str = 'reports'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['critic']
        super().__post_init__()
        return self

    def __str__(self):
        # """Prints to console basic results separate from report."""
        # print('These are the results using the', recipe.model.technique,
        #       'model')
        # if recipe.splicer.technique != 'none':
        #     print('Testing', recipe.splicer.technique, 'predictors')
        # print('Confusion Matrix:')
        # print(self.confusion)
        # print('Classification Report:')
        # print(self.classification_report)
        return self 
 
    def draft(self):
        super().publish()
        self.options = {
            'classification': ['sklearn.metrics', 'classification_report'],
            'confusion': ['sklearn.metrics', 'confusion_matrix']}
        return self

    def publish(self):
        self.runtime_parameters = {
            'y_true': self.recipe.ingredients.y_test,
            'y_pred': self.recipe.predictions}
        super().publish()
        return self
         
