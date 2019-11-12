"""
.. module:: reducer
:synopsis: drops features based upon algorithmic criteria
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleComposer
from simplify.core.technique import SimpleDesign
from simplify.core.decorators import numpy_shield


@dataclass
class Reducer(SimpleComposer):
    """Reduces features through various techniques.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'reducer'

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self.options = {
            'kbest': SimpleDesign(
                name = 'kbest',
                module = 'sklearn.feature_selection',
                algorithm = 'SelectKBest',
                default = {'k': 10, 'score_func': 'f_classif'},
                selected = True),
            'fdr': SimpleDesign(
                name = 'fdr',
                module = 'sklearn.feature_selection',
                algorithm = 'SelectFdr',
                default = {'alpha': 0.05, 'score_func': 'f_classif'},
                selected = True),
            'fpr': SimpleDesign(
                name = 'fpr',
                module = 'sklearn.feature_selection',
                algorithm = 'SelectFpr',
                default = {'alpha': 0.05, 'score_func': 'f_classif'},
                selected = True),
            'custom': SimpleDesign(
                name = 'custom',
                module = 'sklearn.feature_selection',
                algorithm = 'SelectFromModel',
                default = {'threshold': 'mean'},
                runtime = {'estimator': 'estimator'},
                selected = True),
            'rank': SimpleDesign(
                name = 'rank',
                module = 'simplify.critic.rank',
                algorithm = 'RankSelect',
                selected = True),
            'rfe': SimpleDesign(
                name = 'rfe',
                module = 'sklearn.feature_selection',
                algorithm = 'RFE',
                default = {'n_features_to_select': 10, 'step': 1},
                runtime = {'estimator': 'estimator'},
                selected = True),
            'rfecv': SimpleDesign(
                name = 'rfecv',
                module = 'sklearn.feature_selection',
                algorithm = 'RFECV',
                default = {'n_features_to_select': 10, 'step': 1},
                runtime = {'estimator': 'estimator'},
                selected = True)}

#        self.scorers = {'f_classif': f_classif,
#                        'chi2': chi2,
#                        'mutual_class': mutual_info_classif,
#                        'mutual_regress': mutual_info_regression}
        return self

    @numpy_shield
    def publish(self, ingredients, plan = None, estimator = None):
        if not estimator:
            estimator = plan.model.algorithm
        self._set_parameters(estimator)
        self.algorithm = self.options[self.technique](**self.parameters)
        if len(ingredients.x_train.columns) > self.num_features:
            self.algorithm.fit(ingredients.x_train, ingredients.y_train)
            mask = ~self.algorithm.get_support()
            ingredients.drop_columns(df = ingredients.x_train, mask = mask)
            ingredients.drop_columns(df = ingredients.x_test, mask = mask)
        return ingredients

    @numpy_shield
    def publish(self, ingredients: Ingredients, data_to_use: str,
                columns: list = None, **kwargs):
        """[summary]

        Args:
            ingredients (Ingredients): [description]
            data_to_use (str): [description]
            columns (list, optional): [description]. Defaults to None.
        """
        if self.technique != 'none':
            if self.data_dependents:
                self._add_data_dependents(ingredients = ingredients)
            if self.hyperparameter_search:
                self.algorithm = self._search_hyperparameters(
                    ingredients = ingredients,
                    data_to_use = data_to_use)
            try:
                self.algorithm.fit(
                    X = getattr(ingredients, ''.join(['x_', data_to_use])),
                    Y = getattr(ingredients, ''.join(['y_', data_to_use])),
                    **kwargs)
                setattr(ingredients, ''.join(['x_', data_to_use]),
                        self.algorithm.transform(X = getattr(
                            ingredients, ''.join(['x_', data_to_use]))))
            except AttributeError:
                try:
                    ingredients = self.algorithm.publish(
                        ingredients = ingredients,
                        data_to_use = data_to_use,
                        columns = columns,
                        **kwargs)
        return ingredients

    # def _set_parameters(self, estimator):
#        if self.technique in ['rfe', 'rfecv']:
#            self.default = {'n_features_to_select': 10,
#                                       'step': 1}
#            self.runtime_parameters = {'estimator': estimator}
#        elif self.technique == 'kbest':
#            self.default = {'k': 10,
#                                       'score_func': f_classif}
#            self.runtime_parameters = {}
#        elif self.technique in ['fdr', 'fpr']:
#            self.default = {'alpha': 0.05,
#                                       'score_func': f_classif}
#            self.runtime_parameters = {}
#        elif self.technique == 'custom':
#            self.default = {'threshold': 'mean'}
#            self.runtime_parameters = {'estimator': estimator}
#        self._publish_parameters()
#        self._select_parameters()
#        self.parameters.update({'estimator': estimator})
#        if 'k' in self.parameters:
#            self.num_features = self.parameters['k']
#        else:
#            self.num_features = self.parameters['n_features_to_select']
        # return self

