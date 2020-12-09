"""
.. module:: critic
:synopsis: model evaluation made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass
from dataclasses.dataclasses import dataclasses.field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simplify.core.library import Book
from simplify.core.library import Chapter
from simplify.core.library import Technique
from simplify.core.manager import Worker
from simplify.core.repository import SimpleRepository
from simplify.core.scholar import Finisher
from simplify.core.scholar import Parallelizer
from simplify.core.scholar import Scholar
from simplify.core.scholar import Specialist


@dataclasses.dataclass
class Anthology(Book):
    """Applies techniques to 'Cookbook' instances to assess performance.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'anthology'
        chapters (Optional[List['Chapter']]): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty list.
        iterable(Optional[str]): name of property to store alternative proxy
            to 'reviews'.
        steps (Optional[List[Tuple[str, str]]]): tuples of steps and
            techniques.
        techniques (Optional[List['Technique']]): 'Technique' instances to
            apply. In an ordinary project, 'techniques' are not passed to an
            Anthology instance, but are instead created from 'steps' when the
            'publish' method of a 'Project' instance is called. Defaults to
            an empty list.

    """
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'anthology')
    chapters: Optional[List['Review']] = dataclasses.field(default_factory = list)
    iterable: Optional[str] = dataclasses.field(default_factory = lambda: 'reviews')
    steps: Optional[List[Tuple[str, str]]] = dataclasses.field(default_factory = list)
    techniques: Optional[List['Technique']] = dataclasses.field(default_factory = list)


@dataclasses.dataclass
class Review(Chapter):
    """Evaluations for a 'Cookbook' recipe.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.
        steps (Optional[List[str]]):
        explanations (Dict[str, pd.DataFrame]): results from any 'Explainer'
            methods applied to the data analysis. Defaults to an empty
            dictionary.
        predictions (Dict[str, pd.Series]): results from any 'Predictor'
            methods applied to the data analysis. Defaults to an empty
            dictionary.
        estimations (Dict[str, pd.Series]): results from any 'Estimator'
            methods applied to the data analysis. Defaults to an empty
            dictionary.
        importances (Dict[str, pd.DataFrame]): results from any 'Ranker'
            methods applied to the data analysis. Defaults to an empty
            dictionary.
        reports (Dict[str, pd.DataFrame]): results from any 'Reporter'
            methods applied to the data analysis. Defaults to an empty
            dictionary.

    """
    name: Optional[str] = None
    steps: Optional[List[str]] = dataclasses.field(default_factory = list)
    explanations: Optional[Dict[str, pd.DataFrame]] = dataclasses.field(
        default_factory = dict)
    predictions: Optional[Dict[str, pd.Series]] = dataclasses.field(
        default_factory = dict)
    importances: Optional[Dict[str, pd.DataFrame]] = dataclasses.field(
        default_factory = dict)
    metrics: Optional[Dict[str, pd.Series]] = dataclasses.field(
        default_factory = dict)
    reports: Optional[Dict[str, pd.DataFrame]] = dataclasses.field(
        default_factory = dict)


@dataclasses.dataclass
class Evaluator(Technique):
    """Base method wrapper for applying algorithms to data.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        algorithm (Optional[object]): process object which executes the primary
            method of a class instance. Defaults to None.

    """
    name: Optional[str] = None
    module: Optional[str] = None
    algorithm: Optional[object] = None

    """ Private Methods """

    def _get_estimator(self, chapter: 'Chapter') -> 'Technique':
        """Gets 'model' 'Technique' from a list of 'steps' in 'chapter'.

        Args:
            chapter ('Chapter'): instance with 'model' step.

        Returns:
            'Technique': with a 'step' of 'model'.

        """
        for technique in chapter.techniques:
            if technique.step in ['model']:
                return technique
                break
            else:
                pass

    def _get_algorithm(self, estimator: object) -> object:
        algorithm = self.options[self.algorithm_types[estimator.name]]
        return algorithm.load('algorithm')

    """ Core siMpLify Methods """

    def apply(self, recipe: 'Recipe') -> 'Review':
        return self.algorithm.apply(recipe = recipe)


@dataclasses.dataclass
class CriticScholar(Scholar):
    """Applies an 'Anthology' instance to an applied 'Cookbook'.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self = self.idea.apply(instance = self)
        # Creates 'Finisher' instance to finalize 'Technique' instances.
        self.finisher = CriticFinisher(worker = self.worker)
        # Creates 'Specialist' instance to apply 'Technique' instances.
        self.specialist = CriticSpecialist(worker = self.worker)
        # Creates 'Parallelizer' instance to apply 'Chapter' instances, if the
        # option to parallelize has been selected.
        if self.parallelize:
            self.parallelizer = Parallelizer(idea = self.idea)
        return self


@dataclasses.dataclass
class CriticFinisher(Finisher):
    """Finalizes 'Technique' instances with data-dependent parameters.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

    """ Private Methods """

    def _add_explain_conditionals(self,
            technique: 'Technique',
            data: 'Dataset') -> 'Technique':
        """Adds any conditional parameters to 'technique'

        Args:
            technique ('Technique'): an instance with 'algorithm' and
                'parameters' not yet combined.
            data ('Dataset'): data object used to derive hyperparameters.

        Returns:
            'Technique': with any applicable parameters added.

        """

        return technique


@dataclasses.dataclass
class CriticSpecialist(Specialist):
    """Base class for applying 'Technique' instances to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

    """ Private Methods """

    def _get_data(self, data: 'Dataset') -> 'DataBunch':
        """Returns 'data' for model evaluation.

        Args:
            data ('Dataset'): primary instance used by 'project'.

        Returns:
            'DataBunch': with data to use in model evaluation.

        """
        return getattr(data, self.idea['critic']['data_to_review'])

    """ Core siMpLify Methods """

    def apply(self, book: 'Book', data: Union['Dataset', 'Book']) -> 'Book':
        """Applies 'Book' instance in 'project' to 'data' or other stored books.

        Args:
            book ('Book'): instance with stored 'Technique' instances (either
                stored in the 'techniques' or 'chapters' attributes).
            data ([Union['Dataset', 'Book']): a data source with information to
                finalize 'parameters' for each 'Technique' instance in 'book'

        Returns:
            'Book': with 'parameters' for each 'Technique' instance finalized
                and connected to 'algorithm'.

        """
        return super().apply(book = book, data = self._get_data(data = data))


@dataclasses.dataclass
class Evaluators(SimpleRepository):
    """A dictonary of Evaluator options for the Analyst subpackage.

    Args:
        idea (Optional[Idea]): shared 'Idea' instance with project settings.

    """
    idea: Optional[core.Idea] = None

    """ Private Methods """

    def create(self) -> None:
        self.contents = {
            'explain': {
                'eli5': Evaluator(
                    name = 'eli5_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'Eli5Explain'),
                'shap': Evaluator(
                    name = 'shap_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'ShapExplain'),
                'skater': Evaluator(
                    name = 'skater_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'SkaterExplain'),
                'sklearn': Evaluator(
                    name = 'sklearn_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'SklearnExplain')},
            'predict': {
                'eli5': Evaluator(
                    name = 'eli5_predict',
                    module = 'simplify.critic.predictors',
                    algorithm = 'Eli5Predict'),
                'shap': Evaluator(
                    name = 'shap_predict',
                    module = 'simplify.critic.predictors',
                    algorithm = 'ShapPredict'),
                'skater': Evaluator(
                    name = 'skater_predict',
                    module = 'simplify.critic.predictors',
                    algorithm = 'SkaterPredict'),
                'sklearn': Evaluator(
                    name = 'sklearn_predict',
                    module = 'simplify.critic.predictors',
                    algorithm = 'SklearnPredict')},
            'rank': {
                'eli5': Evaluator(
                    name = 'eli5_rank',
                    module = 'simplify.critic.rankers',
                    algorithm = 'Eli5Rank'),
                'shap': Evaluator(
                    name = 'shap_rank',
                    module = 'simplify.critic.rankers',
                    algorithm = 'ShapRank'),
                'skater': Evaluator(
                    name = 'skater_rank',
                    module = 'simplify.critic.rankers',
                    algorithm = 'SkaterRank'),
                'sklearn': Evaluator(
                    name = 'sklearn_rank',
                    module = 'simplify.critic.rankers',
                    algorithm = 'SklearnRank')},
            'measure': {
                'simplify': Evaluator(
                    name = 'simplify_measure',
                    module = 'simplify.critic.metrics',
                    algorithm = 'SimplifyMeasure'),
                'sklearn': Evaluator(
                    name = 'sklearn_measure',
                    module = 'simplify.critic.metrics',
                    algorithm = 'SklearnMeasure')},
            'report': {
                'simplify': Evaluator(
                    name = 'simplify_report',
                    module = 'simplify.critic.reporters',
                    algorithm = 'SimplifyReport'),
                'sklearn': Evaluator(
                    name = 'sklearn_report',
                    module = 'simplify.critic.reporters',
                    algorithm = 'SklearnReport')}}
        return self


@dataclasses.dataclass
class Critic(Worker):
    """Object construction instructions used by a Project instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        book (Optional[str]): name of Book object in 'module' to load. Defaults
            to 'Book'.
        chapter (Optional[str]): name of Chapter object in 'module' to load.
            Defaults to 'Chapter'.
        technique (Optional[str]): name of Book object in 'module' to load.
            Defaults to 'Technique'.
        publisher (Optional[str]): name of Publisher class in 'module' to load.
            Defaults to 'Publisher'.
        scholar (Optional[str]): name of Scholar class in 'module' to load.
            Defaults to 'Scholar'.
        steps (Optional[List[str]]): list of steps to execute. Defaults to an
            empty list.
        options (Optional[Union[str, Dict[str, Any]]]): a dictionary containing
            options for the 'Worker' instance to utilize or a string
            corresponding to a dictionary in 'module' to load. Defaults to an
            empty dictionary.
        data (Optional[str]): name of attribute or key in a 'Project' instance
            'books' to use as a data object to apply methods to. Defaults to
            'dataset'.
        import_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.
        idea (Optional[Idea]): shared project configuration settings.

    """
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'critic')
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.critic.critic')
    book: Optional[str] = dataclasses.field(default_factory = lambda: 'Anthology')
    chapter: Optional[str] = dataclasses.field(default_factory = lambda: 'Review')
    technique: Optional[str] = dataclasses.field(default_factory = lambda: 'Evaluator')
    scholar: Optional[str] = dataclasses.field(default_factory = lambda: 'CriticScholar')
    options: Optional[str] = dataclasses.field(default_factory = lambda: 'Evaluators')
    data: Optional[str] = dataclasses.field(default_factory = lambda: 'analyst')
    idea: Optional[core.Idea] = None

    """ Core siMpLify Methods """

    def outline(self) -> Dict[str, List[str]]:
        """Creates dictionary with techniques for each step.

        Returns:
            Dict[str, Dict[str, List[str]]]: dictionary with keys of steps and
                values of lists of techniques.

        """
        catalog = {}
        steps = self._get_settings(
            section = self.name,
            prefix = self.name,
            suffix = 'steps')
        for step in steps:
            techniques = self._get_settings(
                section = self.name,
                prefix = self.name,
                suffix = 'techniques')
            catalog[step] = []
            for technique in techniques:
                if technique in self.options:
                    catalog[step].append(technique)
        return catalog
    
    
"""
.. module:: critic algorithms
:synopsis: siMpLify algorithms for project evaluation
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass
from dataclasses.dataclasses import dataclasses.field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.critic.critic import Evaluator


def compute_metrics(data: 'Dataset') -> 'Dataset':
    return data

def pandas_describe(data: 'Dataset') -> 'Dataset':
    return data

def simplify_report(data: 'Dataset') -> 'Dataset':
    return data

# eli5_explanation = Technique(
#     name = 'eli5_explanation',
#     module = 'eli5',
#     algorithm = 'explain_prediction_df')


# """ Prediction SimpleRepository """

# prediction_gini = Technique(
#     name = 'gini_predictions',
#     module = 'self',
#     algorithm = '_get_gini_predictions')
# prediction_shap = Technique(
#     name = 'shap_predictions',
#     module = 'self',
#     algorithm = '_get_shap_predictions')

# """ Probability Teachniques """

# probability_gini = Technique(
#     name = 'gini_predicted_probabilities',
#     module = 'self',
#     algorithm = '_get_gini_probabilities')
# probability_log = Technique(
#     name = 'log_predicted_probabilities',
#     module = 'self',
#     algorithm = '_get_log_probabilities')
# probability_shap = Technique(
#     name = 'shap_predicted_probabilities',
#     module = 'self',
#     algorithm = '_get_shap_probabilities')

# """ Ranking SimpleRepository """

# rank_gini = Technique(
#     name = 'gini_importances',
#     module = 'self',
#     algorithm = '_get_gini_importances')
# rank_eli5 = Technique(
#     name = 'eli5_importances',
#     module = 'self',
#     algorithm = '_get_permutation_importances')
# rank_permutation = Technique(
#     name = 'permutation_importances',
#     module = 'self',
#     algorithm = '_get_eli5_importances')
# rank_shap = Technique(
#     name = 'shap_importances',
#     module = 'self',
#     algorithm = '_get_shap_importances')

""" Metrics SimpleRepository """

# def _get_brier_score_loss_parameters(self, parameters, recipe = None):
#     if self.step in 'brier_score_loss':
#         parameters = {
#             'y_true': getattr(recipe.dataset,
#                                 'y_' + self.data_to_review),
#             'y_prob': recipe.probabilities[:, 1]}
#     elif self.step in ['roc_auc']:
#             parameters = {
#                 'y_true': getattr(recipe.dataset,
#                                 'y_' + self.data_to_review),
#                 'y_score': recipe.probabilities[:, 1]}
#     return parameters

# metrics_accuracy = Technique(
#     name = 'accuracy',
#     module = 'sklearn.metrics',
#     algorithm = 'accuracy_score')
# metrics_adjusted_mutual_info = Technique(
#     name = 'adjusted_mutual_info_score',
#     module = 'sklearn.metrics',
#     algorithm = 'adjusted_mutual_info')
# metrics_adjusted_rand = Technique(
#     name = 'adjusted_rand',
#     module = 'sklearn.metrics',
#     algorithm = 'adjusted_rand_score')
# metrics_balanced_accuracy = Technique(
#     name = 'balanced_accuracy',
#     module = 'sklearn.metrics',
#     algorithm = 'balanced_accuracy_score')
# metrics_brier_score_loss = Technique(
#     name = 'brier_score_loss',
#     module = 'sklearn.metrics',
#     algorithm = 'brier_score_loss')
# metrics_calinski = Technique(
#     name = 'calinski_harabasz',
#     module = 'sklearn.metrics',
#     algorithm = 'calinski_harabasz_score')
# metrics_davies = Technique(
#     name = 'davies_bouldin',
#     module = 'sklearn.metrics',
#     algorithm = 'davies_bouldin_score')
# metrics_completeness = Technique(
#     name = 'completeness',
#     module = 'sklearn.metrics',
#     algorithm = 'completeness_score')
# metrics_contingency_matrix = Technique(
#     name = 'contingency_matrix',
#     module = 'sklearn.metrics',
#     algorithm = 'cluster.contingency_matrix')
# metrics_explained_variance = Technique(
#     name = 'explained_variance',
#     module = 'sklearn.metrics',
#     algorithm = 'explained_variance_score')
# metrics_f1 = Technique(
#     name = 'f1',
#     module = 'sklearn.metrics',
#     algorithm = 'f1_score')
# metrics_f1_weighted = Technique(
#     name = 'f1_weighted',
#     module = 'sklearn.metrics',
#     algorithm = 'f1_score',
#     required = {'average': 'weighted'})
# metrics_fbeta = Technique(
#     name = 'fbeta',
#     module = 'sklearn.metrics',
#     algorithm = 'fbeta_score',
#     required = {'beta': 1})
# metrics_fowlkes = Technique(
#     name = 'fowlkes_mallows',
#     module = 'sklearn.metrics',
#     algorithm = 'fowlkes_mallows_score')
# metrics_hamming = Technique(
#     name = 'hamming_loss',
#     module = 'sklearn.metrics',
#     algorithm = 'hamming_loss')
# metrics_h_completness = Technique(
#     name = 'homogeneity_completeness',
#     module = 'sklearn.metrics',
#     algorithm = 'homogeneity_completeness_v_measure')
# metrics_homogeniety = Technique(
#     name = 'homogeneity',
#     module = 'sklearn.metrics',
#     algorithm = 'homogeneity_score')
# metrics_jaccard = Technique(
#     name = 'jaccard_similarity',
#     module = 'sklearn.metrics',
#     algorithm = 'jaccard_similarity_score')
# metrics_mae = Technique(
#     name = 'median_absolute_error',
#     module = 'sklearn.metrics',
#     algorithm = 'median_absolute_error')
# metrics_matthews_corrcoef = Technique(
#     name = 'matthews_correlation_coefficient',
#     module = 'sklearn.metrics',
#     algorithm = 'matthews_corrcoef')
# metrics_max_error = Technique(
#     name = 'max_error',
#     module = 'sklearn.metrics',
#     algorithm = 'max_error')
# metrics_mean_absolute_error = Technique(
#     name = 'mean_absolute_error',
#     module = 'sklearn.metrics',
#     algorithm = 'mean_absolute_error')
# metrics_mse = Technique(
#     name = 'mean_squared_error',
#     module = 'sklearn.metrics',
#     algorithm = 'mean_squared_error')
# metrics_msle = Technique(
#     name = 'mean_squared_log_error',
#     module = 'sklearn.metrics',
#     algorithm = 'mean_squared_log_error')
# metrics_mutual_info = Technique(
#     name = 'mutual_info_score',
#     module = 'sklearn.metrics',
#     algorithm = 'mutual_info_score')
# metrics_log_loss = Technique(
#     name = 'log_loss',
#     module = 'sklearn.metrics',
#     algorithm = 'log_loss')
# metrics_norm_mutual_info = Technique(
#     name = 'normalized_mutual_info',
#     module = 'sklearn.metrics',
#     algorithm = 'normalized_mutual_info_score')
# metrics_precision = Technique(
#     name = 'precision',
#     module = 'sklearn.metrics',
#     algorithm = 'precision_score')
# metrics_precision_weighted = Technique(
#     name = 'precision_weighted',
#     module = 'sklearn.metrics',
#     algorithm = 'precision_score',
#     required = {'average': 'weighted'})
# metrics_r2 = Technique(
#     name = 'r2',
#     module = 'sklearn.metrics',
#     algorithm = 'r2_score')
# metrics_recall = Technique(
#     name = 'recall',
#     module = 'sklearn.metrics',
#     algorithm = 'recall_score')
# metrics_recall_weighted = Technique(
#     name = 'recall_weighted',
#     module = 'sklearn.metrics',
#     algorithm = 'recall_score',
#     required = {'average': 'weighted'})
# metrics_roc_auc = Technique(
#     name = 'roc_auc',
#     module = 'sklearn.metrics',
#     algorithm = 'roc_auc_score')
# metrics_silhouette = Technique(
#     name = 'silhouette',
#     module = 'sklearn.metrics',
#     algorithm = 'silhouette_score')
# metrics_v_measure = Technique(
#     name = 'v_measure',
#     module = 'sklearn.metrics',
#     algorithm = 'v_measure_score')
# metrics_zero_one = Technique(
#     name = 'zero_one',
#     module = 'sklearn.metrics',
#     algorithm = 'zero_one_loss')


# @dataclasses.dataclass
# class Article(object):

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     """ Private Methods """

#     def _add_row(self, recipe, report):
#         new_row = pd.Series(index = self.columns)
#         for column, variable in self.required_columns.items():
#             new_row[column] = getattr(recipe, variable)
#         for column in report:
#             new_row[column] = report[column]
#         self.text.loc[len(self.text)] = new_row
#         return self

#     def _check_best(self, recipe):
#         """Checks if the current recipe is better than the current best recipe
#         based upon the primary scoring metric.

#         Args:
#             recipe: an instance of Recipe to be tested versus the current best
#                 recipe stored in the 'best_recipe' attribute.
#         """
#         if not self._exists('best_recipe'):
#             self.best_recipe = recipe
#             self.best_recipe_score = self.article.loc[
#                     self.article.index[-1],
#                     utilities.listify(self.metrics)[0]]
#         elif (self.article.loc[
#                 self.article.index[-1],
#                 utilities.listify(self.metrics)[0]] > self.best_recipe_score):
#             self.best_recipe = recipe
#             self.best_recipe_score = self.article.loc[
#                     self.article.index[-1],
#                     utilities.listify(self.metrics)[0]]
#         return self

#     def _format_step(self, attribute):
#         if getattr(self.recipe, attribute).step in ['none', 'all']:
#             step_column = getattr(self.recipe, attribute).step
#         else:
#             step = getattr(self.recipe, attribute).step
#             parameters = getattr(self.recipe, attribute).parameters
#             step_column = f'{step}, parameters = {parameters}'
#         return step_column

#     def _get_step_name(self, step):
#         """Returns appropriate algorithm to the report attribute."""
#         if step.step in ['none', 'all']:
#             return step.step
#         else:
#             return step.algorithm

#     def print_best(self):
#         """Prints output to the console about the best recipe."""
#         if self.verbose:
#             print('The best test recipe, based upon the',
#                   utilities.listify(self.metrics)[0], 'metric with a score of',
#                   f'{self.best_recipe_score: 4.4f}', 'is:')
#             for step in getattr(self,
#                     self.iterator).best_recipe.steps:
#                 print(step.capitalize(), ':',
#                       getattr(getattr(self, self.iterator).best_recipe,
#                               step).step)
#         return

#     def _set_columns(self, recipe):
#         self.required_columns = {
#             'recipe_number': 'number',
#             'options': 'steps',
#             'seed': 'seed',
#             'validation_set': 'using_val_set'}
#         self.columns = list(self.required_columns.keys())
#         self.columns.extend(recipe.steps)
#         for step in self.steps:
#             if (hasattr(getattr(self, step), 'columns')
#                     and getattr(self, step).name != 'summarize'):
#                 self.columns.extend(getattr(self, step).columns)
#         return self

#     def _start_report(self, recipe):
#         self._set_columns(recipe = recipe)
#         self.text = pd.DataFrame(columns = self.columns)
#         return self

#     """ Public Import/Export Methods """

#     def save(self, report = None):
#         """Exports the review report to disk.

#         Args:
#             review(Review.report): 'report' from an instance of review
#         """
#         self.filer.save(
#             variable = report,
#             folder = self.filer.experiment,
#             file_name = self.model_type + '_review',
#             file_format = 'csv',
#             header = True)
#         return

#     """ Core siMpLify Methods """

#     def draft(self) -> None:
#         super().draft()
#         return self

#     def publish(self):
#         super().publish()
#         return self




# DEFAULT_OPTIONS = {
#     'accuracy': ['sklearn.metrics', 'accuracy_score'],
#     'adjusted_mutual_info': ['sklearn.metrics', 'adjusted_mutual_info_score'],
#     'adjusted_rand': ['sklearn.metrics', 'adjusted_rand_score'],
#     'balanced_accuracy': ['sklearn.metrics', 'balanced_accuracy_score'],
#     'brier_score_loss': ['sklearn.metrics', 'brier_score_loss'],
#     'calinski': ['sklearn.metrics', 'calinski_harabasz_score'],
#     'davies': ['sklearn.metrics', 'davies_bouldin_score'],
#     'completeness': ['sklearn.metrics', 'completeness_score'],
#     'contingency_matrix': ['sklearn.metrics', 'cluster.contingency_matrix'],
#     'explained_variance': ['sklearn.metrics', 'explained_variance_score'],
#     'f1': ['sklearn.metrics', 'f1_score'],
#     'f1_weighted': ['sklearn.metrics', 'f1_score'],
#     'fbeta': ['sklearn.metrics', 'fbeta_score'],
#     'fowlkes': ['sklearn.metrics', 'fowlkes_mallows_score'],
#     'hamming': ['sklearn.metrics', 'hamming_loss'],
#     'h_completness': ['sklearn.metrics', 'homogeneity_completeness_v_measure'],
#     'homogeniety': ['sklearn.metrics', 'homogeneity_score'],
#     'jaccard': ['sklearn.metrics', 'jaccard_similarity_score'],
#     'mae': ['sklearn.metrics', 'median_absolute_error'],
#     'matthews_corrcoef': ['sklearn.metrics', 'matthews_corrcoef'],
#     'max_error': ['sklearn.metrics', 'max_error'],
#     'mean_absolute_error': ['sklearn.metrics', 'mean_absolute_error'],
#     'mse': ['sklearn.metrics', 'mean_squared_error'],
#     'msle': ['sklearn.metrics', 'mean_squared_log_error'],
#     'mutual_info': ['sklearn.metrics', 'mutual_info_score'],
#     'neg_log_loss': ['sklearn.metrics', 'log_loss'],
#     'norm_mutual_info': ['sklearn.metrics', 'normalized_mutual_info_score'],
#     'precision': ['sklearn.metrics', 'precision_score'],
#     'precision_weighted': ['sklearn.metrics', 'precision_score'],
#     'r2': ['sklearn.metrics', 'r2_score'],
#     'recall': ['sklearn.metrics', 'recall_score'],
#     'recall_weighted': ['sklearn.metrics', 'recall_score'],
#     'roc_auc': ['sklearn.metrics', 'roc_auc_score'],
#     'silhouette': ['sklearn.metrics', 'silhouette_score'],
#     'v_measure': ['sklearn.metrics', 'v_measure_score'],
#     'zero_one': ['sklearn.metrics', 'zero_one_loss']}


# @dataclasses.dataclass
# class Metrics(Evaluator):
#     """Measures model performance.

#     Args:
#         step(str): name of step.
#         parameters(dict): dictionary of parameters to pass to selected
#             algorithm.
#         name(str): designates the name of the class which is used throughout
#             siMpLify to match methods and settings with this class and
#             identically named subclasses.
#         auto_draft(bool): whether 'publish' method should be called when
#             the class is instanced. This should generally be set to True.

#     """
#     step: object = None
#     parameters: object = None
#     name: str = 'metrics'
#     auto_draft: bool = True
#     options: Dict = dataclasses.field(default_factory = lambda: DEFAULT_OPTIONS)

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     def _build_conditional_parameters(self, parameters, recipe = None):
#         if self.step in 'brier_score_loss':
#             parameters = {
#                 'y_true': getattr(recipe.dataset,
#                                   'y_' + self.data_to_review),
#                 'y_prob': recipe.probabilities[:, 1]}
#         elif self.step in ['roc_auc']:
#              parameters = {
#                  'y_true': getattr(recipe.dataset,
#                                    'y_' + self.data_to_review),
#                  'y_score': recipe.probabilities[:, 1]}
#         return parameters

#     def draft(self) -> None:
#         super().draft()
#         self.negative_options = [
#             'brier_loss_score',
#             'neg_log_loss',
#             'zero_one']
#         self.required = {
#             'fbeta': {'beta': 1},
#             'f1_weighted': {'average': 'weighted'},
#             'precision_weighted': {'average': 'weighted'},
#             'recall_weighted': {'average': 'weighted'}}
#         return self

#     # def edit(self, name, metric, special_type = None,
#     #          special_parameters = None, negative_metric = False):
#     #     """Allows user to manually add a metric to report."""
#     #     self.workers.update({name: metric})
#     #     if special_type in ['probability']:
#     #         self.prob_options.update({name: metric})
#     #     elif special_type in ['scorer']:
#     #         self.score_options.update({name: metric})
#     #     if special_parameters:
#     #        self.special_options.update({name: special_parameters})
#     #     if negative_metric:
#     #        self.negative_options.append[name]
#     #     return self

#     def publish(self, recipe):
#         self.runtime_parameters = {
#             'y_true': getattr(recipe.dataset, 'y_' + self.data_to_review),
#             'y_pred': recipe.predictions}
#         super().implement()
#         return self


    # def _get_gini_probabilities(self, recipe):
    #     """Estimates probabilities of outcomes from fitted model with gini
    #     method.

    #     Args:
    #         recipe(Recipe): instance of Recipe with a fitted model.

    #     Returns:
    #         Series with predictions from fitted model on test data.

    #     """
    #     if hasattr(recipe.model.algorithm, 'predict_proba'):
    #         return recipe.model.algorithm.predict_proba(
    #             getattr(recipe.dataset, 'x_' + self.data_to_review))[1]
    #     else:
    #         if self.verbose:
    #             print('predict_proba method does not exist for',
    #                   recipe.model.step.name)
    #         return None

    # def _get_log_probabilities(self, recipe):
    #     """Estimates log probabilities of outcomes from fitted model.

    #     Args:
    #         recipe(Recipe): instance of Recipe with a fitted model.

    #     Returns:
    #         Series with predictions from fitted model on test data.

    #     """
    #     if hasattr(recipe.model.algorithm, 'predict_log_proba'):
    #         return recipe.model.algorithm.predict_log_proba(
    #             getattr(recipe.dataset, 'x_' + self.data_to_review))[1]
    #     else:
    #         if self.verbose:
    #             print('predict_log_proba method does not exist for',
    #                   recipe.model.step.name)
    #         return None

    # def _get_permutation_importances(self, recipe):
    #     scorer = utilities.listify(self.metrics_steps)[0]
    #     base_score, score_decreases = self.workers[self.step](
    #             score_func = scorer,
    #             x = getattr(recipe.dataset, 'x_' + self.data_to_review),
    #             y = getattr(recipe.dataset, 'y_' + self.data_to_review))
    #     return np.mean(score_decreases, axis = 'columns')

    # def _get_gini_importances(self, recipe):
    #     features = list(getattr(
    #             recipe.dataset, 'x_' + self.data_to_review).columns)
    #     if hasattr(recipe.model.algorithm, 'feature_importances_'):
    #         importances = pd.Series(
    #             data = recipe.model.algorithm.feature_importances_,
    #             index = features)
    #         return importances.sort_values(ascending = False, inplace = True)
    #     else:
    #         return None

    # def _get_shap_importances(self, recipe):
    #     if hasattr(recipe, 'shap_explain.values'):
    #         return np.abs(recipe.shap_explain.values).mean(0)
    #     else:
    #         return None

    # def _get_eli5_importances(self, recipe):
    #     base_score, score_decreases = get_score_importances(score_func, X, y)
    #     feature_importances = np.mean(score_decreases, axis=0)
    #     from eli5 import show_weights
    #     self.permutation_weights = show_weights(
    #             self.permutation_importances,
    #             feature_names = recipe.dataset.columns.keys())
    #     return self