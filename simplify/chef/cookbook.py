"""
.. module:: cookbook
:synopsis: data analysis and machine learning builder module
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from itertools import product

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
    steps: object = None
    name: str = 'chef'
    auto_publish: bool = True
    auto_implement: bool = False
    lazy_import: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _publish_recipes(self):
        for i, plan in enumerate(self.plans):
            steps = {}
            for j, technique in enumerate(plan):
                steps.update({self.sequence[j]: technique})
            self.recipes.update(
                    {str(i + 1): Recipe(number = i + 1, steps = steps)})
        return self

    def _implement_recipes(self, using_val_set = False):
        """Tests all 'recipes'."""
        for number, recipe in self.recipes.items():
            if self.verbose:
                print('Testing', recipe.name, str(recipe.number))
            recipe.using_val_set = using_val_set
            recipe.implement(ingredients = self.ingredients)
            if self.export_results:
                self.depot._set_experiment_folder()
                self.depot._set_plan_folder(iterable = recipe, 
                                            name = 'recipe')
                if self.export_all_recipes:
                    self.save_recipes(recipes = recipe)
                if 'reduce' in self.sequence and recipe.reduce != 'none':
                    self.ingredients.save_dropped(folder = self.depot.recipe)
                else:
                    self.ingredients.save_dropped(
                        folder = self.depot.experiment)
            # if 'critic' in self.packages:
            #     self.critic.implement(ingredients = recipe.ingredients,
            #                           recipes = recipe)
            # if 'artist' in self.packages:
            #     self.artist.implement(ingredients = self.critic.ingredients,
            #                           recipes = recipe,
            #                           reviews = self.critic.reviews)
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
                recipes = self.recipes
            for recipe in recipes:
                self.depot._set_recipe_folder(recipe = recipe)
                recipe.save(folder = self.depot.recipe)
        elif recipes in ['best'] and hasattr(self, 'critic'):
            self.critic.best_recipe.save(file_path = file_path,
                                         folder = self.depot.experiment,
                                         file_name = 'best_recipe')
        elif not isinstance(recipes, str):
            recipes.save(file_path = file_path, folder = self.depot.recipe)
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
        if self.recipes is None:
            self.recipes = {}
        self.comparer = True
        self.depot.step = 'chef'
        return self

    def edit_recipes(self, recipes):
        """Adds a single recipe or list of recipes to 'recipes' attribute.

        Args:
            recipes(Recipe, list(Recipe), or dict(int, Recipe)): recipes to be
                added into 'recipes' attribute.
        """
        if self.recipes is None:
            setattr(self, self.iterator, {})
        if recipes:
            if isinstance(recipes, dict):
                recipes = list(recipes.values())
                last_num = list(self.recipes.keys())[-1:]
            else:
                last_num = 0
            for i, recipe in enumerate(self.listify(recipes)):
                self.recipes.update({last_num + i + 1: recipe})
        return self

    def publish(self):
        Recipe.options = self.options
        Recipe.sequence = self.sequence
        self._publish_recipes()
        return self

    def implement(self, ingredients = None, previous_package = None):
        """Completes an iteration of a Cookbook.

        Args:
            ingredients(Ingredients): If passsed, it will be assigned to the
                local 'ingredients' attribute. If not passed, and if it already
                exists, the local 'ingredients' will be used.
            previous_package(SimpleIterable): The previous subpackage, if one
                was used

        """
        if ingredients:
            self.ingredients = ingredients
        if previous_package:
            self.ingredients = previous_package.ingredients
        if 'train_test_val' in self.data_to_use:
            self.ingredients._remap_dataframes(data_to_use = 'train_test')
            self._implement_recipes()
            self.ingredients._remap_dataframes(data_to_use = 'train_val')
            self._implement_recipes(using_val_set = True)
        else:
            self.ingredients._remap_dataframes(data_to_use = self.data_to_use)
            self._implement_recipes()
        if self.export_results:
            self.save_recipes(recipes = 'best')
        return self


@dataclass
class Recipe(SimpleIterable):
    """Contains steps for analyzing data in the siMpLify Cookbook subpackage.

    Args:
        number(int): number of recipe in a sequence - used for recordkeeping
            purposes.
        steps(dict): dictionary containing keys of SimpleTechnique names
            (strings) and values of SimpleIterable subclass instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    number: int = 0
    steps: object = None
    name: str = 'recipe'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['chef']
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
        if self.steps['model'] in ['xgboost']:
            # Model class is injected with scale_pos_weight for algorithms that
            # use that parameter.
            self.model.scale_pos_weight = (
                    len(self.ingredients.y.index) /
                    ((self.ingredients.y == 1).sum())) - 1
        return self

    """ Public Import/Export Methods """

    def save(self, file_path = None, folder = None, file_name = None):
        self.depot.save(variable = self,
                        file_path = file_path,
                        folder = folder,
                        file_name = file_name,
                        file_format = 'pickle')
        return

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        if not self.options:
            self.options = {
                'scale': ['simplify.chef.steps.scale', 'Scale'],
                'split': ['simplify.chef.steps.split', 'Split'],
                'encode': ['simplify.chef.steps.encode', 'Encode'],
                'mix': ['simplify.chef.steps.mix', 'Mix'],
                'cleave': ['simplify.chef.steps.cleave', 'Cleave'],
                'sample': ['simplify.chef.steps.sample', 'Sample'],
                'reduce': ['simplify.chef.steps.reduce', 'Reduce'],
                'model': ['simplify.chef.steps.model', 'Model']}
        self.sequence_setting = 'chef_steps'
        return self

    def implement(self, ingredients):
        """Applies the recipe steps to the passed ingredients."""
        sequence = self.sequence.copy()
        self.ingredients = ingredients
        self.ingredients.split_xy(label = self.label)
        if self._calculate_hyperparameters:
            self._calculate_hyperparameters            
        # If using cross-validation or other data splitting technique, the
        # pre-split methods apply to the 'x' data. After the split, steps
        # must incorporate the split into 'x_train' and 'x_test'.
        for step in self.sequence:
            sequence.remove(step)
            if step == 'split':
                break
            else:
                self.ingredients = getattr(self, step).implement(
                    ingredients = self.ingredients,
                    plan = self)
        for train_index, test_index in self.split.algorithm.split(
                self.ingredients.x, self.ingredients.y):
            self.ingredients.x_train, self.ingredients.x_test = (
                   self.ingredients.x.iloc[train_index],
                   self.ingredients.x.iloc[test_index])
            self.ingredients.y_train, self.ingredients.y_test = (
                   self.ingredients.y.iloc[train_index],
                   self.ingredients.y.iloc[test_index])
            for step in sequence:
                self.ingredients = getattr(self, step).implement(
                       ingredients = self.ingredients,
                       plan = self)
        return self