"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and utilization.
"""
from dataclasses import dataclass
from itertools import product

from .recipe import Recipe
from .steps import Cleave
from .steps import Encode
from .steps import Mix
from .steps import Model
from .steps import Reduce
from .steps import Sample
from .steps import Scale
from .steps import Split
from ..critic import Critic
from ..implements import listify
from ..managers import Planner, Step


@dataclass
class Cookbook(Planner):
    """Dynamically creates recipes for preprocessing, machine learning, and
        data analysis using a unified interface and architecture.

    Attributes:
        menu: an instance of Menu.
        inventory: an instance of Inventory. If one is not passed when Cookbook
            is instanced, one will be created with default options.
        steps: a dictionary containing strings as keys and classes as values.
            If steps is not passed when Cookbook is instanced, the default
            steps will be used. These steps can still be modified or
            supplemented with Cookbook methods.
        ingredients: an instance of Ingredients.
        recipes: a list of instances of Recipe which Cookbook creates through
            the prepare method and applies through the create method.
        auto_prepare: sets whether to automatically call the prepare method
            when the class is instanced. If you do not plan to make any
            adjustments to the steps, techniques, or algorithms beyond the
            menu, this option should be set to True. If you plan to make such
            changes, prepare should be called when those changes are complete.
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
    """
    menu : object
    inventory : object = None
    steps : object = None
    ingredients : object = None
    recipes : object = None
    auto_prepare : bool = True
    name : str = 'cookbook'

    def __post_init__(self):
        """Sets up the core attributes of Cookbook."""
        if self.menu['general']['verbose']:
            print('Creating cookbook')
        # Declares default step names and classes in a cookbook recipe.
        self.default_steps = {'scaler' : Scale,
                              'splitter' : Split,
                              'encoder' : Encode,
                              'mixer' : Mix,
                              'cleaver' : Cleave,
                              'sampler' : Sample,
                              'reducer' : Reduce,
                              'model' : Model}
        super().__post_init__()
        return self

    def _check_best(self, recipe):
        """Checks if the current Recipe is better than the current best Recipe
        based upon the primary scoring metric.
        """
        if not self.best_recipe:
            self.best_recipe = recipe
            self.best_recipe_score = self.critic.review.report.loc[
                    self.critic.review.report.index[-1],
                    listify(self.metrics)[0]]
        elif (self.critic.review.report.loc[
                self.critic.review.report.index[-1],
                listify(self.metrics)[0]] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.critic.review.report.loc[
                    self.critic.review.report.index[-1],
                    listify(self.metrics)[0]]
        return self

    def _check_steps(self):
        """Checks if steps currently exist. If not, default_steps are used."""
        if not self.steps:
            self.steps = self.default_steps
        return self

    def _compute_hyperparameters(self):
        """Computes hyperparameters that can be determined by the source data.
        """
        # Data is split in oder for certain values to be computed that require
        # features and the label to be split.
        self.ingredients.split_xy(label = self.label)
        Model.scale_pos_weight = (len(self.ingredients.y.index) /
                                  ((self.ingredients.y == 1).sum())) - 1
        return self

    def _localize(self):
        """Injects parameters and 'general' section of menu into each step
        base class and relevant menu settings into this class instance."""
        for step, step_class in self.steps.items():
            if (hasattr(step_class, 'name')
                    and step_class.name in self.menu.config):
                if not step_class.parameters:
                    setattr(step_class, 'parameters', self.menu.config[step])
            self.menu.localize(instance = step_class, sections = ['general'])
        self.menu.localize(instance = Recipe, sections = ['general'])
        sections = ['general', 'files']
        if hasattr(self, 'name') and self.name in self.menu.config:
            sections.append(self.name)
        self.menu.localize(instance = self, sections = sections)
        return self

    def _prepare_steps(self):
        """Initializes the step classes for use by the Cookbook."""
        self.step_lists = []
        self._check_steps()
        self._localize()
        for step in self.steps.keys():
            setattr(self, step, listify(getattr(self, step)))
            self.step_lists.append(getattr(self, step))
        # Creates a list of all possible permutations of step techniques
        # selected. Each item in the the list is a 'plan'
        self.all_recipes = list(map(list, product(*self.step_lists)))
        return self

    def _start_recipes(self, recipes = None, data_to_use = 'train_test'):
        """Completes one iteration of a Cookbook, storing the review in the
        review report dataframe. Plots and the recipe are exported to the
        recipe folder.
        """
        for recipe in recipes:
            if self.verbose:
                print('Testing recipe ' + str(recipe.number))
            self.inventory._set_recipe_folder(
                    recipe = recipe, steps_to_use = ['model', 'cleaver'])
            self.ingredients.split_xy(label = self.label)
            recipe.start(ingredients = self.ingredients,
                         data_to_use = data_to_use)
            self.critic.start(recipe = recipe)
            self._check_best(recipe)
            self.save_classification_report()
            self.ingredients._remap_dataframes(data_to_use = 'train_test')
            # To conserve memory, each recipe is deleted after being exported.
            del(recipe)
        return self

    def add_cleave(self, cleave_group, prefixes = [], columns = []):
        """Adds cleaves to the list of cleaves."""
        if not hasattr(self.cleaves) or not self.cleaves:
            self.cleaves = []
        columns = self.ingredients.create_column_list(prefixes = prefixes,
                                                      columns = columns)
        Cleave.add(cleave_group = cleave_group, columns = columns)
        self.cleaves.append(cleave_group)
        return self

    def add_parameters(self, step, parameters):
        """Adds parameter sets to the parameters dictionary of a prescribed
        step. """
        self.steps[step].add_parameters(parameters = parameters)
        return self

    def add_recipe(self, recipe):
        if hasattr(self, 'recipes'):
            self.recipes.append(recipe)
        else:
            self.recipes = [recipe]
        return self

    def add_runtime_parameters(self, step, parameters):
        """Adds runtime_parameter sets to the parameters dictionary of a
        prescribed step."""
        self.steps[step].add_runtime_parameters(parameters = parameters)
        return self

    def add_step(self, name, techniques, step_order = None, **kwargs):
        self.steps.update({name : Step(name = name,
                                       techniques = techniques,
                                       **kwargs)})
        if step_order:
            self.set_order(order = step_order)
        return self

    def add_techniques(self, step, techniques, algorithms):
        """Adds new technique name and corresponding algorithm to the
        techniques dictionary.
        """
        self.steps[step].add_techniques(techniques = techniques,
                                        algorithms = algorithms)
        return self

    def load_recipe(self, file_path):
        """Imports a single recipe from disc."""
        recipe = self.inventory.unpickle_object(file_path)
        self.add_recipe(recipe = recipe)
        return self

    def prepare(self):
        """Creates a planner with all possible selected permutations of
        methods. Each set of methods is stored in a list of instances of the
        class stored in self.recipes.
        """
        # Injects menu instance into Model base class to allow Model to access
        # specific model parameters.
        Model.menu = self.menu
        # Creates Critic instance to compile results and create visualizations.
        self.critic = Critic(menu = self.menu, inventory = self.inventory)
        # If option is selected, select hyperparameters are computed if they
        # can be derived from the data (without creating endogeniety problems).
        if self.compute_hyperparameters:
            self._compute_hyperparameters()
        self._prepare_steps()
        self.recipes = []
        self.custom_steps = {}
        for i, recipe in enumerate(self.all_recipes):
            recipe_params = {'number' : i + 1, 'order' : self.order}
            for j, step in enumerate(self.steps.keys()):
                if step in self.default_steps:
                    recipe_params.update({step : self.steps[step](recipe[j])})
                else:
                    self.custom_steps.update(
                            {step : self.steps[step](recipe[j])})
            self.recipes.append(Recipe(**recipe_params))
            for step, step_class in self.custom_steps.items():
                setattr(self.recipes[-1], step, step_class)
        return self

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  listify(self.metrics)[0], 'metric with a score of',
                  f'{self.best_recipe_score : 4.4f}', 'is:')
            for step in self.best_recipe.order:
                print(step.capitalize(), ':',
                      getattr(self.best_recipe, step).technique)
        return

    def save(self):
        """Exports the list of recipes to disc as one object."""
        cookbook_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = 'cookbook.pkl')
        self.inventory.pickle_object(self.recipes, file_path = cookbook_path)
        return self

    def save_all_recipes(self, recipes = None):
        if not recipes:
            recipes = self.recipes
        for recipe in recipes:
            file_name = (
                'recipe' + str(recipe.number) + '_' + recipe.model.technique)
            recipe_path = self.inventory.create_path(
                    folder = self.inventory.recipe,
                    file_name = file_name,
                    file_type = 'pickle')
            self.save_recipe(recipe = recipe, file_path = recipe_path)
        return

    def save_best_recipe(self):
        if hasattr(self, 'best_recipe'):
            best_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = 'best_recipe.pkl')
            self.inventory.pickle_object(self.best_recipe,
                                         file_path = best_path)
        return self

    def save_classification_report(self, classification_report = None):
        if not classification_report:
            classification_report = self.critic.review.classification_report_df
        file_path = self.inventory.create_path(
                    folder = self.inventory.recipe,
                    file_name = 'class_report',
                    file_type = 'csv')
        self.inventory.save_df(classification_report, file_path = file_path)
        return

    def save_everything(self):
        """Automatically saves the recipes, results, dropped columns from
        ingredients, and the best recipe (if one has been stored)."""
        self.save()
        self.save_review()
        self.save_best_recipe()
        self.ingredients.save_drops()
        return self

    def save_recipe(self, recipe, file_path):
        """Exports a recipe to disc."""
        self.inventory.pickle_object(recipe, file_path)
        return self

    def save_review(self):
        review_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = 'review.csv')
        self.inventory.save_df(self.critic.review.report,
                               file_path = review_path)
        return self

    def set_order(self, order = None):
        """Sets order to default (order of steps.keys) if one is not provided.
        """
        if order:
            self.order = order
        elif hasattr(self, 'steps') and self.steps:
            self.order = list(self.steps.keys())
        return self

    def start(self, recipes = None, data_to_use = 'train_test'):
        """Iterates through each of the possible recipes. The best overall
        recipe is stored in self.best_recipe. If the user has selected both
        the testing and validation sets to be tested, all recipes are applied
        with both sets of data.
        """
        if self.verbose:
            print('Testing recipes')
        if not recipes:
            recipes = self.recipes
        self.best_recipe = None
        if self.data_to_use == 'train_test_val':
            self._start_recipes(recipes = recipes, data_to_use = 'train_test')
            self._start_recipes(recipes = recipes, data_to_use = 'train_val')
        else:
            self._start_recipes(recipes = recipes,
                                data_to_use = self.data_to_use)
        if self.export_all_recipes:
            self.save_all_recipes()
        return self