"""
.. module:: cookbook
:synopsis: data analysis and machine learning builder module
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.package import SimplePackage
from simplify.core.package import SimplePlan


@dataclass
class Cookbook(SimplePackage):
    """Dynamically creates recipes for staging, machine learning, and data
    analysis using a unified interface and architecture.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        recipes (Recipe or list(Recipe)): Ordinarily, 'recipes' is not passed
            when Cookbook is instanced, but the argument is included if the
            user wishes to reexamine past recipes or manually create new
            recipes.
        techniques (dict): keys are names of techniques to be applied and 
            values are the siMpLify-compatible python objects applying those 
            techniques.

    Since this class is a subclass to SimplePackage and SimpleClass, 
    documentation for those classes applies as well.

    """
    name: str = 'chef'
    recipes: object = None
    techniques: object = None

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """

    def _extra_processing(self, plan: SimpleClass,
            data: SimpleClass) -> Tuple[SimpleClass, SimpleClass]:
        """Tests all 'recipes'."""
        recipe.using_val_set = self.using_val_set
        if self.export_results:
            self.depot._set_experiment_folder()
            self.depot._set_plan_folder(
                plan = plan,
                name = 'recipe')
            if self.export_all_recipes:
                self.save_recipes(recipes = plan)
            if 'reduce' in self.order and plan.steps['reduce'] != 'none':
                data.save_dropped(folder = self.depot.recipe)
            else:
                data.save_dropped(folder = self.depot.experiment)
        return plan, data

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

    """ Public Import/Export Methods """

    def load_recipe(self, file_path):
        """Imports a single recipe from disc and adds it to the class iterable.

        Args:
            file_path: a path where the file to be loaded is located.
        """
        self.load_plan(file_path = file_path)
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
            'scaler': ('simplify.chef.techniques.scale', 'Scale'),
            'splitter': ('simplify.chef.techniques.split', 'Split'),
            'encoder': ('simplify.chef.techniques.encode', 'Encode'),
            'mixer': ('simplify.chef.techniques.mix', 'Mix'),
            'cleaver': ('simplify.chef.techniques.cleave', 'Cleave'),
            'sampler': ('simplify.chef.techniques.sample', 'Sample'),
            'reducer': ('simplify.chef.techniques.reduce', 'Reduce'),
            'modeler': ('simplify.chef.techniques.model', 'Model')}
        # Locks 'stage' attribute at 'chef' for state dependent methods.
        self.stage.change('chef')
        return self

    def edit_recipes(self, recipes):
        """Adds a recipe or list of recipes to 'recipes' attribute.
        Args:
            recipes (dict(str/int: Recipe or list(dict(str/int: Recipe)):
                recipe(s) to be added to 'recipes'.
                
        """
        self.edit_plans(plans = recipes)
        return self

    def publish(self, ingredients = None):
        """Completes an iteration of a Cookbook.

        Args:
            ingredients (Ingredients): If passed, it will be assigned to the
                local 'ingredients' attribute. If not passed, and if it already
                exists, the local 'ingredients' will be used.

        """
        if ingredients:
            self.ingredients = ingredients
        if 'train_test_val' in self.data_to_use:
            self.ingredients.state = 'train_test'
            ingredients = super().publish(data = self.ingredients)
            self.ingredients.state = 'train_val'
            ingredients = super().publish(data = self.ingredients)
        elif 'full' in self._data_to_use:
            self.ingredients.state = 'full'
            ingredients = super().publish(data = self.ingredients)
        else:
            self.ingredients.state = 'train_test'
            ingredients = super().publish(data = self.ingredients)
        if self.export_results:
            self.save_recipes(recipes = 'best')
        return self

    """ Properties """
    
    @property
    def comparer(self):
        return Recipe
    
    @comparer.setter
    def comparer(self, comparer: SimplePlan):
        self.comparer = comparer
        return self
    
    @property
    def recipes(self):
        return self.plans

    @recipes.setter
    def recipes(self, plans: dict):
        self.plans = plans
        return self


@dataclass
class Recipe(SimplePlan):
    """Contains techniques for analyzing data in the siMpLify Cookbook subpackage.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        number (int): number of plan in a sequence - used for recordkeeping
            purposes.
        steps (dict(str: str)): keys are names of steps and values are 
            algorithms to be applied.

    It is also a child class of SimpleClass and Simple. So, documentation for
    those classes applies as well.

    """
    name: str = 'recipe'
    number: int = 0
    steps: object = None

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
        if self.techniques['model'] in ['xgboost']:
            # Model class is injected with scale_pos_weight for algorithms that
            # use that parameter.
            self.model.scale_pos_weight = (
                    len(self.ingredients.y.index) /
                    ((self.ingredients.y == 1).sum())) - 1
        return self

    """ Core siMpLify Methods """