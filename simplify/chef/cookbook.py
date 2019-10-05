"""
.. module:: cookbook
:synopsis: data analysis and machine learning builder module
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
import datetime

from simplify.chef.recipe import Recipe
from simplify.core.decorators import local_backups
from simplify.core.iterables import SimpleBuilder
from simplify.core.technique import SimpleTechnique


@dataclass
class Cookbook(SimpleBuilder):
    """Dynamically creates recipes for staging, machine learning, and data
    analysis using a unified interface and architecture.

    Args:
        ingredients(Ingredients or str): an instance of Ingredients or a string
            with the file path for a pandas DataFrame that will. This argument
            does not need to be passed when the class is instanced. However,
            failing to do so will prevent the use of the Cleave step and the
           '_calculate_hyperparameters' method. 'ingredients' will need to be
            passed to the 'implement' method if it isn't when the class is
            instanced. Consequently, it is recommended that 'ingredients' be
            passed when the class is instanced.
        steps(dict(str: SimpleStep)): steps to be completed in order. This
            argument should only be passed if the user wishes to override the
            steps listed in the Idea settings or if the user is not using the
            Idea class.
        recipes(Recipe or list(Recipe)): Ordinarily, 'recipes' is not passed
            when Cookbook is instanced, but the argument is included if the user
            wishes to reexamine past recipes or manually create new recipes.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced. If you do not plan to make any
            adjustments to the steps, techniques, or algorithms beyond the
            Idea configuration, this option should be set to True. If you plan
            to make such changes, 'publish' should be called when those
            changes are complete.
        auto_implement(bool): whether to call the 'implement' method when the 
            class is instanced.

    Since this class is a subclass to SimpleBuilder and SimpleClass, all
    documentation for those classes applies as well.

    """

    ingredients: object = None
    steps: object = None
    recipes: object = None
    name: str = 'chef'
    auto_publish: bool = True
    auto_implement: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _calculate_hyperparameters(self):
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
            self.options['model'].scale_pos_weight = (
                    len(self.ingredients.y.index) /
                    ((self.ingredients.y == 1).sum())) - 1
        return self
    
    def _implement_recipes(self):
        """Tests 'recipes' with all combinations of step techniques selected.
        """
        for recipe_number, recipe in getattr(self, self.iterable).items():
            if self.verbose:
                print('Testing', recipe.name, str(recipe_number))
            recipe.implement(ingredients = self.ingredients)
            if self.export_all_recipes:
                self.save_recipe(recipe = recipe)
            if 'critic' in self.packages:
                self.critic.implement(ingredients = recipe.ingredients,
                                      recipes = recipe)
            if 'artist' in self.packages:
                self.artist.implement(ingredients = self.critic.ingredients,
                                      recipes = recipe,
                                      reviews = self.critic.reviews)
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

        Args:
            plan: an instance of Almanac or Recipe for which files are to be
                saved.
            steps to use: a list of strings or single string containing names
                of steps from which the folder name should be created.
        """
        if hasattr(self, 'naming_classes') and self.naming_classes is not None:
            subfolder = 'recipe_'
            for step in self.listify(self.naming_classes):
                subfolder += recipe.steps[step].technique + '_'
            subfolder += str(recipe.number)
            self.depot.recipe = self.depot.create_folder(
                folder = self.depot.experiment, subfolder = subfolder)
        return self

    """ Public Tool Methods """

    def add_cleaves(self, cleave_group, prefixes = None, columns = None):
        """Adds cleaves to the list of cleaves.

        Args:
            cleave_group: string naming the set of features in the group.
            prefixes: list or string of prefixes for columns to be included
                within the cleave.
            columns: list or string of columns to be included within the
                cleave."""
        if not hasattr(self, 'cleaves') or self.cleaves is None:
            self.cleaves = []
        columns = self.ingredients.create_column_list(prefixes = prefixes,
                                                      columns = columns)
        self.options['cleave'].edit(cleave_group = cleave_group,
                    columns = columns)
        self.cleaves.append(cleave_group)
        return self

    def load_recipe(self, file_path):
        """Imports a single recipe from disc and adds it to self.recipes.

        Args:
            file_path: a path where the file to be loaded is located.
        """
        self.edit_recipes(recipe = self.depot.load(file_path = file_path,
                                                   file_format = 'pickle'))
        return self

    def print_best(self):
        """Calls critic instance print_best method. The method is added here
        for easier accessibility.
        """
        self.analysis.print_best()
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
        self.save_best_recipe()
        self.ingredients.save_dropped()
        return

    def save_recipe(self, recipe, file_path = None):
        """Exports a recipe to disc.

        Args:
            recipe: an instance of Recipe.
            file_path: path of where file should be saved. If none, a default
                file_path will be created from self.depot."""
        if self.verbose:
            print('Saving recipe', recipe.number)
        self._set_recipe_folder(recipe = recipe)
        self.depot.save(variable = recipe,
                        file_path = file_path,
                        folder = self.depot.recipe,
                        file_name = 'recipe',
                        file_format = 'pickle')
        return

    """ Core siMpLify Methods """

    def draft(self):
        """Sets default options for the Chef's cookbook."""
        super().draft()
        self.options = {
                'scale': ['simplify.chef.steps.scale', 'Scale'],
                'split': ['simplify.chef.steps.split', 'Split'],
                'encode': ['simplify.chef.steps.encode', 'Encode'],
                'mix': ['simplify.chef.steps.mix', 'Mix'],
                'cleave': ['simplify.chef.steps.cleave', 'Cleave'],
                'sample': ['simplify.chef.steps.sample', 'Sample'],
                'reduce': ['simplify.chef.steps.reduce', 'Reduce'],
                'model': ['simplify.chef.steps.model', 'Model']}
        # Adds GPU check to other checks to be implemented.
        self.checks.extend(['gpu', 'ingredients'])
        # Locks 'step' attribute at 'cook' for conform methods in package.
        self.step = 'cook'
        # Sets attributes to allow proper parent methods to be used.
        self.iterable_type = 'parallel'
        self.iterable = 'recipes'
        self.iterable_class = Recipe
        self.iterable_setting = 'cookbook_steps'
        self.return_variables = None
        return self

    def edit_recipes(self, recipes):
        """Adds a single recipe or list of recipes to 'recipes' attribute.

        Args:
            recipes(Recipe, list(Recipe), or dict(int, Recipe)): recipes to be
                added into 'recipes' attribute.
        """
        if self.recipes is None:
            self.recipes = {}
        if self.recipes:
            if isinstance(recipes, dict):
                recipes = list(recipes.values())
                last_num = list(self.recipes.keys())[-1:]
            for i, recipe in enumerate(self.listify(recipes)):
                self.recipes.update({last_num + i + 1: recipe})
        elif isinstance(recipes, dict):
            self.recipes = recipes
        else:
            self.recipes = {}
            for i, recipe in enumerate(self.listify(recipes)):
                self.recipes.update({i + 1: recipe})
        return self

    def publish(self):
        """Creates a planner with all possible selected permutations of
        methods. Each set of methods is stored in a list of instances of the
        class stored in self.recipes.
        """
        self._set_experiment_folder()
        # Creates all recipe combinations and store Recipe instances in
        # 'recipes'.
        super().publish()
        if 'critic' in self.packages:
            from simplify.critic.review import Review
            print('instancing critic')
            self.critic = Review()
        if 'artist' in self.packages:
            from simplify.artist.canvas import Canvas
            self.artist = Canvas()
        return self

#    @local_backups
    def implement(self, ingredients = None):
        """Completes an iteration of a Cookbook.

        Args:
            ingredients: an Instance of Ingredients. If passsed, it will be
                assigned to self.ingredients. If not passed, and if it already
                exists, self.ingredients will be used.
        """
        if ingredients:
            self.ingredients = ingredients
        if 'train_test_val' in self.data_to_use:
            self.ingredients._remap_dataframes(data_to_use = 'train_test')
            self._implement_recipes()
            self.ingredients._remap_dataframes(data_to_use = 'train_val')
            self._implement_recipes()
        else:
            self.ingredients._remap_dataframes(data_to_use = self.data_to_use)
            self._implement_recipes()
        return self
