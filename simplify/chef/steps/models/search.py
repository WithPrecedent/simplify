
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from ....core.technique import Technique


@dataclass
class Search(Technique):
    
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'search_parameters'

    def __post_init__(self):
        self.search_parameters = self.idea[self.name]
        super().__post_init__()
        return self

    def draft(self):
        self.options = {'random' : RandomizedSearchCV,
                        'grid' : GridSearchCV,
                        'bayes' : BayesSearchCV}
        self.runtime_parameters = {
            'estimator' : self.parameters['estimator'],
            'param_distributions' : self.parameters['space'],
            'random_state' : self.seed}
        self.checks = ['idea']
        return self

    def finalize(self):
        if 'refit' in self.search_parameters:
            self.search_parameters['scoring'] = self.listify(
                    self.search_parameters['scoring'])[0]
        self.search_parameters.update(self.runtime_parameters)
        self.algorithm = self.options[self.technique](**self.search_parameters)
        return self

    def produce(self, ingredients):
        if self.verbose:
            print('Searching for best hyperparameters using',
                  self.technique, 'search algorithm')
        self.algorithm.fit(ingredients.x_train, ingredients.y_train)
        self.best_estimator = self.algorithm.best_estimator_
        if self.verbose:
            print('The', self.search_parameters['scoring'],
                  'score of the best estimator for this model is',
                  f'{self.algorithm.best_score_ : 4.4f}')
        return self