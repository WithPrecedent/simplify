
from dataclasses import dataclass

from scipy.stats import randint, uniform

from simplify.chef.steps.models.search import Search
from simplify.core.technique import Technique


@dataclass
class Algorithm(Technique):

    def __post_init__(self):
        super().__post_init__()
        return self

    def _edit_specific_parameters(self):
        if (hasattr(self, 'model_parameters')
                and self.technique in self.model_parameters):
           self.parameters.update(self.model_parameters[self.technique])
        if (self.technique in ['xgboost']
                and not 'scale_pos_weight' in self.parameters
                and hasattr(self, 'scale_pos_weight')):
            self.parameters.update(
                    {'scale_pos_weight' : self.scale_pos_weight})
        return self

    def _gpu_parameters(self):
        if self.technique in ['xgboost']:
            self.runtime_parameters.update({'tree_method' : 'gpu_exact'})
        return self

    def _list_type(self, test_list, data_type):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, data_type) for i in test_list)

    def _parse_parameters(self):
        """Parses parameters to determine if the user has created ranges of
        parameters. If so, parameters are divided between those to be searched
        and those that are fixed. If any parameters include a range of values,
        hyperparameter_search is set to True. Fixed parameters are stored in
        the parameters attribute. Ranges to be searched are stored in the
        space attribute.
        """
        self.hyperparameter_search = False
        self.space = {}
        new_parameters = {}
        for param, values in self.parameters.items():
            if isinstance(values, list):
                self.hyperparameter_search = True
                if self._list_type(values, float):
                    self.space.update({param : uniform(values[0], values[1])})
                elif self._list_type(values, int):
                    self.space.update({param : randint(values[0], values[1])})
            else:
                new_parameters.update({param : values})
        self.parameters = new_parameters
        return self

    def _finalize_search(self):
        """Instances and finalizes search technique class."""
        self.searcher = Search(
                technique = self.idea['cookbook']['search_algorithm'],
                estimator = self.algorithm,
                parameters = self.search_parameters,
                space = self.space,
                seed = self.seed,
                verbose = self.verbose)
        self.searcher.finalize()
        return self

    def finalize(self):
        self._edit_specific_parameters()
        if self.gpu:
           self._gpu_parameters()
        self._parse_parameters()
        if self.hyperparameter_search:
            self._finalize_search()
        self.tool = self.options[self.technique](**self.parameters)
        return self

    def produce(self, ingredients):
        if self.hyperparameter_search:
            self.searcher.produce(ingredients = ingredients)
            self.tool = self.searcher.best_estimator
        else:
            self.tool.fit(ingredients.x_train, ingredients.y_train)
        return ingredients