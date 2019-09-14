"""
cookbook.py is the primary control file for the siMpLify machine learning
package. It contains the Cookbook class, which handles the cookbook
construction and utilization.
"""
from dataclasses import dataclass
import datetime
from itertools import product

from simplify.chef.recipe import Recipe
from simplify.chef.steps import (Cleave, Encode, Mix, Model, Reduce, Sample,
                                     Scale, Split)
from simplify.core.base import SimpleClass
from simplify.core.tools import check_arguments
from simplify.critic.critic import Critic


@dataclass
class Cookbook(SimpleClass):
    """Dynamically creates recipes for staging, machine learning, and data
    analysis using a unified interface and architecture.

    Parameters:
        ingredients: an instance of Ingredients (or a subclass). This argument
            does not need to be passed when the class is instanced. However,
            failing to do so will prevent the use of the Cleave step and the
            _compute_hyperparameters method. 'ingredients' will need to be 
            passed to the 'produce' method if it isn't when the class is
            instanced. Consequently, it is recommended that 'ingredients' be
            passed when the class is instanced.
        steps: a list of string step names to be completed in order. This 
            argument should only be passed if the user wishes to override the 
            steps listed in the Idea settings or if the user is not using the
            Idea class.
        recipes: a list of instances of Recipe which Cookbook creates through
            the 'finalize' method and applies through the 'produce' method.
            Ordinarily, 'recipes' is not passed when Cookbook is instanced, but
            the argument is included if the user wishes to reexamine past
            recipes or manually create new recipes.
        name: a string designating the name of the class which should be
            identical to the section of the Idea section with relevant settings.
        auto_finalize: sets whether to automatically call the 'finalize' method
            when the class is instanced. If you do not plan to make any
            adjustments to the steps, techniques, or algorithms beyond the
            Idea configuration, this option should be set to True. If you plan
            to make such changes, 'finalize' should be called when those changes
            are complete.
        auto_produce: sets whether to automatically call the 'produce' method
            when the class is instanced.
    """
    ingredients : object = None
    steps : object = None
    recipes : object = None
    name : str = 'cookbook'
    auto_finalize : bool = True
    auto_produce : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _compute_hyperparameters(self):
        """Computes hyperparameters that can be determined by the source data
        (without creating data leakage problems).
        
        This method currently only support xgboost's scale_pos_weight
        parameter. Future hyperparameter computations will be added as they
        are discovered.
        """
        # 'ingredients' attribute is required before method can be called.
        if self.ingredients is not None:
            # Data is split in oder for certain values to be computed that
            # require features and the label to be split.
            self.ingredients.split_xy(label = self.label)
            # Model class is injected with scale_pos_weight for algorithms that
            # use that parameter.
            Model.scale_pos_weight = (len(self.ingredients.y.index) /
                                    ((self.ingredients.y == 1).sum())) - 1
        return self

    def _finalize_one_loop(self, data_to_use):
        """Prepares one set of recipes from all_recipes as applied to a
        specific training/testing set.

        Parameters:
            data_to_use: a string corresponding to an Ingredients property
                which will return the appropriate training/testing set.
        """
        for i, recipe in enumerate(self.all_recipes):
            recipe_instance = Recipe(techniques = self.steps)
            # Adds number of recipe to recipe instance for differentiation.
            setattr(recipe_instance, 'number', i + 1)
            # Adds data used to recipe instance so that the information is 
            # available for later use, if needed. 
            setattr(recipe_instance, 'data_to_use', data_to_use)
            # Attaches corrresponding technique to recipe instance for a 
            # particular step in the recipe process.
            for j, step in enumerate(self.options.keys()):
                setattr(recipe_instance, step, self.options[step](recipe[j]))
            recipe_instance.finalize()
            self.recipes.append(recipe_instance)
        return self

    def _finalize_recipes(self):
        """Creates all combinations of recipes from user options in the Idea
        instance and stores those in 'all_recipes' attribute.
        """
        self.recipes = []
        step_combinations = []
        for step in self.options.keys():
            # Stores each step attribute in a list
            setattr(self, step, self.listify(getattr(self, step)))
            # Adds step to a list of all step lists
            step_combinations.append(getattr(self, step))
        # Creates a list of all possible permutations of step techniques
        # selected. Each item in the the list is a 'plan'
        self.all_recipes = list(map(list, product(*step_combinations)))
        return self

    def _set_experiment_folder(self):
        """Sets the experiment folder and corresponding attributes in this
        class's Depot instance based upon user settings.
        """
        if self.depot.datetime_naming:
            subfolder = ('experiment_'
                         + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        else:
            subfolder = 'experiment'
        self.depot.experiment = self.depot.create_folder(
                folder = self.depot.results, subfolder = subfolder)
        return self

    def _set_recipe_folder(self, recipe):
        """Creates file or folder path for plan-specific exports.

        Parameters:
            plan: an instance of Almanac or Recipe for which files are to be
                saved.
            steps to use: a list of strings or single string containing names
                of steps from which the folder name should be created.
        """
        if hasattr(self, 'naming_classes') and self.naming_classes is not None:
            subfolder = 'recipe_'
            for step in self.listify(self.naming_classes):
                subfolder += getattr(recipe, step).technique + '_'
            subfolder += str(recipe.number)
            self.depot.recipe = self.depot.create_folder(
                folder = self.depot.experiment, subfolder = subfolder)
        return self

    """ Public Methods """

    def add_cleaves(self, cleave_group, prefixes = None, columns = None):
        """Adds cleaves to the list of cleaves.

        Parameters:
            cleave_group: string naming the set of features in the group.
            prefixes: list or string of prefixes for columns to be included
                within the cleave.
            columns: list or string of columns to be included within the
                cleave."""
        if not hasattr(self, 'cleaves') or self.cleaves is None:
            self.cleaves = []
        columns = self.ingredients.create_column_list(prefixes = prefixes,
                                                      columns = columns)
        Cleave.add(cleave_group = cleave_group, columns = columns)
        self.cleaves.append(cleave_group)
        return self

    def edit_recipes(self, recipes):
        """Adds a single recipe or list of recipes to 'recipes' attribute.

        Parameters:
            recipe: an instance of Recipe.
        """
        if hasattr(self, 'recipes'):
            self.recipes.extend(self.listify(recipes))
        else:
            self.recipes = self.listify(recipes)
        return self

    def load_recipe(self, file_path):
        """Imports a single recipe from disc and adds it to self.recipes.

        Parameters:
            file_path: a path where the file to be loaded is located.
        """
        self.edit_recipes(recipe = self.depot.load(file_path = file_path,
                                                   file_format = 'pickle'))
        return self

    def draft(self):
        """ Declares default step names and plan_class in a Cookbook recipe."""
        # Sets options for default steps of a Recipe.
        self.options = {'scaler' : Scale,
                        'splitter' : Split,
                        'encoder' : Encode,
                        'mixer' : Mix,
                        'cleaver' : Cleave,
                        'sampler' : Sample,
                        'reducer' : Reduce,
                        'model' : Model}
        # Adds GPU check to other checks to be produceed.
        self.checks = ['gpu', 'ingredients', 'steps']
        # Locks 'step' attribute at 'cook' for conform methods in package.
        self.step = 'cook'
        return self

    def finalize(self):
        """Creates a planner with all possible selected permutations of
        methods. Each set of methods is stored in a list of instances of the
        class stored in self.recipes.
        """
        # Adds finalize search_parameters to Model class.
        Model.search_parameters = self.idea['search_parameters']
        # Sets attributes for data analysis and export.
        self.critic = Critic()
        self._set_experiment_folder()
        # Creates all recipe combinations for Idea instance.
        self._finalize_recipes() 
        # Using training, test, validate sets creates two separate loops
        # through all recipes: one with the test set, one with the validation
        # set.
        if 'train_test_val' in self.data_to_use:
            self._finalize_one_loop(data_to_use = 'train_test')
            self._finalize_one_loop(data_to_use = 'train_val')
        else:
            self._finalize_one_loop(data_to_use = self.data_to_use)
        return self

    def print_best(self):
        """Calls critic instance print_best method. The method is added here
        for easier accessibility.
        """
        self.critic.print_best()
        return self
    
    def save_all_recipes(self):
        """Saves all recipes in self.recipes to disc as individual files."""
        for recipe in self.recipes:
            file_name = (
                'recipe' + str(recipe.number) + '_' + recipe.model.technique)
            self.save_recipe(recipe = recipe,
                             folder = self.depot.recipe,
                             file_name = file_name,
                             file_format = 'pickle')
        return

    def save_best_recipe(self):
        """Saves the best recipe to disc."""
        if hasattr(self, 'best_recipe'):
            self.depot.save(variable = self.best_recipe,
                            folder = self.depot.experiment,
                            file_name = 'best_recipe',
                            file_format = 'pickle')
        return

    def save_everything(self):
        """Automatically saves the recipes, results, dropped columns from
        ingredients, and the best recipe (if one has been stored)."""
        self.save()
        self.save_review()
        self.save_best_recipe()
        self.ingredients.save_dropped()
        return

    def save_recipe(self, recipe, file_path = None):
        """Exports a recipe to disc.

        Parameters:
            recipe: an instance of Recipe.
            file_path: path of where file should be saved. If none, a default
                file_path will be created from self.depot."""
        if self.verbose:
            print('Saving recipe', recipe.number)
        self._set_recipe_folder(recipe = recipe)
        self.depot.save(variable = draft,
                        file_path = file_path,
                        folder = self.depot.recipe,
                        file_name = 'recipe',
                        file_format = 'pickle')
        return

    def save_review(self, review = None):
        """Exports the Analysis review to disc.

        Parameters:
            review: the attribute review from an instance of Analysis. If none
                is provided, self.analysis.review is saved.
        """
        if not review:
            review = self.analysis.review.report
        self.depot.save(variable = review,
                        folder = self.depot.experiment,
                        file_name = self.model_type + '_review',
                        file_format = 'csv',
                        header = True)
        return

    @check_arguments
    def produce(self, ingredients = None):
        """Completes an iteration of a Cookbook.

        Parameters:
            ingredients: an Instance of Ingredients. If passsed, it will be
                assigned to self.ingredients. If not passed, and if it already
                exists, self.ingredients will be used.
        """
        if ingredients:
            self.ingredients = ingredients
        for recipe in self.recipes:
            if self.verbose:
                print('Testing ' + recipe.name + ' ' + str(recipe.number))
            recipe.produce(ingredients = ingredients)
            self.save_recipe(recipe = recipe)
            self.analysis.produce(recipe = recipe)
            self._check_best(recipe = recipe)
            # To conserve memory, each recipe is deleted after being exported.
            del(recipe)
        return self