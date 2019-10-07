"""
.. module:: cookbook
:synopsis: data analysis and machine learning builder module
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.iterable import SimpleIterable


@dataclass
class Cookbook(SimpleIterable):
    """Dynamically creates recipes for staging, machine learning, and data
    analysis using a unified interface and architecture.

    Args:
        ingredients(Ingredients, DataFrame, or str): an instance of
            Ingredients, a pandas DataFrame, or a string with the file path for
            data to be loaded into a pandas DataFrame. This argument does not
            need to be passed when the class is instanced. However, failing to
            do so will prevent the use of the Cleave step and the
           '_calculate_hyperparameters' method. 'ingredients' will need to be
            passed to the 'implement' method if it isn't when the class is
            instanced. Consequently, it is recommended that 'ingredients' be
            passed when the class is instanced.
        recipes(Recipe or list(Recipe)): Ordinarily, 'recipes' is not passed
            when Cookbook is instanced, but the argument is included if the
            user wishes to reexamine past recipes or manually create new
            recipes.
        name(str): designates the name of the class which should be identical
            to the section of the Idea instance with relevant settings.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced. If you do not plan to make any adjustments to
            the steps, techniques, or algorithms beyond the Idea configuration,
            this option should be set to True. If you plan to make such
            changes, 'publish' should be called when those changes are
            complete.
        auto_implement(bool): whether to call the 'implement' method when the
            class is instanced.

    Since this class is a subclass to SimpleIterable and SimpleClass, all
    documentation for those classes applies as well.

    """

    ingredients: object = None
    recipes: object = None
    name: str = 'chef'
    auto_publish: bool = True
    auto_implement: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _implement_recipes(self):
        """Tests all 'recipes'."""
        for recipe_number, recipe in getattr(self, self.iterable).items():
            if self.verbose:
                print('Testing', recipe.name, str(recipe_number))
            recipe.implement(ingredients = self.ingredients)
            if self.export_results:
                self.depot._set_experiment_folder()
                self.depot._set_recipe_folder()
                if self.export_all_recipes:
                    self.save_recipes(recipes = recipe)
                if 'reduce' in self.steps and self.reduce != 'none':
                    self.ingredients.save_dropped(folder = self.depot.recipe)
            if 'critic' in self.packages:
                self.critic.implement(ingredients = recipe.ingredients,
                                      recipes = recipe)
            if 'artist' in self.packages:
                self.artist.implement(ingredients = self.critic.ingredients,
                                      recipes = recipe,
                                      reviews = self.critic.reviews)
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

    def print_best(self):
        """Calls critic instance print_best method. The method is added here
        for easier accessibility.
        """
        self.critic.print_best()
        return self

    """ Public Import/Export Methods """

    def load_recipe(self, file_path):
        """Imports a single recipe from disc and adds it to the class iterable.

        Args:
            file_path: a path where the file to be loaded is located.
        """
        self.edit_iterable(iterables = self.depot.load(file_path = file_path,
                                                       file_format = 'pickle'))
        return self

    def save_recipes(self, recipes, file_path = None):
        """Exports a recipe or recipes to disc.

        Args:
            recipe(Recipe, str, list(Recipe)): an instance of Recipe, a list of
                Recipe instances, 'all' (meaning all recipes stored in the
                class iterable), or 'best' (meaning the current best recipe).
            file_path: path of where file should be saved. If none, a default
                file_path will be created from self.depot.

        """
        if recipes in ['all'] or isinstance(recipes, list):
            if recipes in ['all']:
                recipes = getattr(self, self.iterable)
            for recipe in recipes:
                self.depot._set_recipe_folder(recipe = recipe)
                recipe.save(folder = self.depot.recipe)
        elif recipes in ['best']:
            self.critic.best_recipe.save(file_path = file_path,
                                         folder = self.depot.experiment,
                                         file_name = 'best_recipe')
        else:
            recipes.save(file_path = file_path, folder = self.depot.recipe)
        return

    """ Core siMpLify Methods """

    def draft(self):
        """Sets default options for the Chef's cookbook."""
        super().draft()
        self.options = {
            'chef': ['simplify.chef.recipe', 'Recipe'],
            'critic' : ['simplify.critic.review', 'Review'],
            'artist': ['simplify.artist.canvas', 'Canvas']}
        self.parallel_options = {
                'chef': ['scale', 'split', 'encode', 'mix', 'cleave',
                            'sample', 'reduce', 'model']}
        self.checks.extend(['ingredients'])
        # Locks 'step' attribute at 'cook' for conform methods in package.
        self.step = 'cook'
        # Sets attributes to allow proper parent methods to be used.
        self.iterable = 'steps'
        self.iterable_setting = 'packages'
        self.return_variables = {'critic': ['best_recipe']}
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
        if self.export_results:
            self.save_recipes(recipes = 'best')
            if not 'reduce' in self.steps or self.reduce == ['none']:
                self.ingredients.save_dropped(folder = self.depot.experiment)
        return self
