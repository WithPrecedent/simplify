
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from ....core.technique import Technique


@dataclass
class Search(Technique):


    technique : str
    estimator : object
    parameters : object
    space : object
    seed : int
    verbose : bool

    def __post_init__(self):
        super().__post_init__()
        return self

    def plan(self):
        self.options = {'random' : RandomizedSearchCV,
                        'grid' : GridSearchCV,
                        'bayes' : BayesSearchCV}
        self.runtime_parameters = {'estimator' : self.estimator,
                                   'param_distributions' : self.space,
                                   'random_state' : self.seed}
        return self

    def prepare(self):
        if 'refit' in self.parameters:
            self.parameters['scoring'] = self.listify(
                    self.parameters['scoring'])[0]
        self.parameters.update(self.runtime_parameters)
        self.tool = self.options[self.technique](**self.parameters)
        return self

    def perform(self, ingredients):
        if self.verbose:
            print('Searching for best hyperparameters using',
                  self.technique, 'search algorithm')
        self.tool.fit(ingredients.x_train, ingredients.y_train)
        self.best_estimator = self.tool.best_estimator_
        if self.verbose:
            print('The', self.parameters['scoring'],
                  'score of the best estimator for this model is',
                  f'{self.tool.best_score_ : 4.4f}')
        return self