"""
.. module:: almanac
:synopsis: data gathering, munging, and preprocessing builder module
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.decorators import local_backups
from simplify.core.base import (SimpleManager, SimplePlan, SimpleStep,
                                SimpleTechnique)


@dataclass
class Almanac(SimpleManager):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Args:
        idea(Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a 
            supoorted settings file for an Idea instance is located.
        depot(Depot): an instance of Depot.
        ingredients(Ingredients or str): an instance of Ingredients or a string
            with the file path for a pandas DataFrame that will. This argument
            does not need to be passed when the class is instanced. 
        steps(dict(str: SimpleStep)): steps to be completed in order. This
            argument should only be passed if the user wishes to override the
            steps listed in the Idea settings or if the user is not using the
            Idea class.
        plans(SimplePlan): instanced subclasses of SimplePlan for prepared tools
            for the Almanac.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced. If you do not plan to make any
            adjustments to the steps, techniques, or algorithms beyond the
            Idea configuration, this option should be set to True. If you plan
            to make such changes, 'publish' should be called when those
            changes are complete.
        auto_produce(bool): whether to call the 'produce' method when the class
            is instanced.
            
    Since this class is a subclass to SimpleManager and SimpleClass, all
    documentation for those classes applies as well.

    """
    idea: object = None
    depot: object = None
    ingredients: object = None
    steps: object = None
    plans: object = None
    name: str = 'cookbook'
    auto_publish: bool = True
    auto_produce: bool = True

    def __post_init__(self):
        """Sets up the core attributes of Harvest."""
        super().__post_init__()
        return self

    def _check_defaults(self):
        for name in self.__dict__.copy().keys():
            if name.startswith('default_'):
                new_name = name.lstrip('default_')
                if not hasattr(self, new_name):
                    setattr(self, new_name, getattr(self, name))
        return self

    def _check_drafts(self):
        if isinstance(self.drafts, dict):
            for key, value in self.drafts:
                setattr(self, key, value)
        return self

    def _check_sections(self):
        if not hasattr(self, 'sections') or not self.sections:
            if hasattr(self, 'default_sections'):
                self.sections = self.default_sections
            else:
                self.sections = {}
        return self

    def _publish_draft(self):
        """Initializes the step classes for use by the Harvest."""
        self.drafts = []
        for step in self.steps:
            step_instance = self.draft_class(name = step,
                                            index_column = self.index_column)
            for technique in self.listify(getattr(self, step + '_techniques')):
                tool_instance = self.edit_technique(
                        step = step,
                        technique = technique,
                        parameters = self.listify(getattr(self, technique)))
                step_instance.techniques.append(tool_instance)
            step_instance.publish()
            self.drafts.append(step_instance)
        return self

    def _produce_file(self, ingredients):
        with open(
                self.depot.path_in, mode = 'r', errors = 'ignore',
                encoding = self.idea['files']['file_encoding']) as a_file:
            ingredients.source = a_file.read()
            for technique in self.techniques:
                ingredients = technique.produce(ingredients = ingredients)
            self.depot.save(variable = ingredients.df)
        return ingredients

    def _produce_glob(self, ingredients):
        self.depot.initialize_writer(
                file_path = self.depot.path_out)
        ingredients.create_series()
        for file_num, a_path in enumerate(self.depot.path_in):
            if (file_num + 1) % 100 == 0 and self.verbose:
                print(file_num + 1, 'files parsed')
            with open(
                    a_path, mode = 'r', errors = 'ignore',
                    encoding = self.idea['files']['file_encoding']) as a_file:
                ingredients.source = a_file.read()
                print(ingredients.df)
                ingredients.df[self.index_column] = file_num + 1
                for technique in self.techniques:
                    ingredients = technique.produce(ingredients = ingredients)
                self.depot.save(variable = ingredients.df)
        return ingredients

    def _set_columns(self, organizer):
        if not hasattr(self, 'columns'):
            self.columns = {self.index_column: int}
            if self.metadata_columns:
                self.columns.update(self.metadata_columns)
        self.columns.update(dict.fromkeys(self.columns, str))
        return self

    def draft(self):
        """ Declares default step names and classes in an Harvest."""
        super().draft()
        self.options = {
                'sow': ['simplify.farmer.sow', 'Sow'],
                'harvest': ['simplify.farmer.harvest', 'Harvest'],
                'clean': ['simplify.farmer.clean', 'Clean'],
                'bale': ['simplify.farmer.bale', 'Bale'],
                'deliver': ['simplify.farmer.deliver', 'Deliver']}
        self.draft_class = Almanac
        self.checks.extend(['drafts', 'sections', 'defaults'])
        return self

    def publish(self):
        """Creates a Harvest with all sequenced techniques applied at each
        step. Each set of methods is stored in a list within a Almanac instance.
        """
        if self.verbose:
            print('Preparing Harvest')
        self._publish_draft_class()
        self._publish_steps()
        self._publish_draft()
        if hasattr(self, '_set_folders'):
            self._set_folders()
        return self

    def produce(self, ingredients = None):
        """Completes an iteration of an Harvest."""
        if not ingredients:
            ingredients = self.ingredients
        for draft in self.drafts:
            self.step = draft.name
            # Adds initial columns dictionary to ingredients instance.
            if (self.step in ['reap']
                    and 'organize' in self.reap_techniques):
                self._set_columns(organizer = draft)
                ingredients.columns = self.columns
            self.conform(step = self.step)
            self.ingredients = draft.produce(ingredients = self.ingredients)
            self.depot.save(variable = self.ingredients,
                                file_name = self.step + '_ingredients')
        return self