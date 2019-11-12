"""
.. module:: almanac
:synopsis: data gathering, munging, and preprocessing builder module
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.decorators import local_backups
from simplify.core.package import SimplePackage
from simplify.core.technique import FarmerTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleComposite
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleComposite subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'sow': ['simplify.farmer.sow', 'Sow'],
    'harvest': ['simplify.farmer.harvest', 'Harvest'],
    'clean': ['simplify.farmer.clean', 'Clean'],
    'bale': ['simplify.farmer.bale', 'Bale'],
    'deliver': ['simplify.farmer.deliver', 'Deliver']}


@dataclass
class Almanac(SimplePackage):
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
        techniques(dict(str: FarmerTechnique)): techniques to be completed in order. This
            argument should only be passed if the user wishes to override the
            techniques listed in the Idea settings or if the user is not using the
            Idea class.
        plans(SimplePackage): instanced subclasses of SimplePackage for
            prepared tools for the Almanac.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_draft(bool): whether to call the 'publish' method when the
            class is instanced. If you do not plan to make any
            adjustments to the techniques, techniques, or algorithms beyond the
            Idea configuration, this option should be set to True. If you plan
            to make such changes, 'publish' should be called when those
            changes are complete.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.

    Since this class is a subclass to SimplePackage and SimpleComposite, all
    documentation for those classes applies as well.

    """
    idea: object = None
    depot: object = None
    ingredients: object = None
    techniques: object = None
    plans: object = None
    name: str = 'chef'
    auto_draft: bool = True
    auto_publish: bool = True

    def __post_init__(self) -> None:
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

    def _publish_draft(self) -> None:
        """Initializes the step classes for use by the Harvest."""
        self.drafts = []
        for step in self.techniques:
            step_instance = self.draft_class(name = step,
                                            index_column = self.index_column)
            for technique in listify(getattr(self, step + '_techniques')):
                tool_instance = self.edit_technique(
                        step = step,
                        technique = technique,
                        parameters = listify(getattr(self, technique)))
                step_instance.techniques.append(tool_instance)
            step_instance.publish()
            self.drafts.append(step_instance)
        return self

    def _implement_file(self, ingredients):
        with open(
                self.depot.path_in, mode = 'r', errors = 'ignore',
                encoding = self.idea['files']['file_encoding']) as a_file:
            ingredients.source = a_file.implement()
            for technique in self.techniques:
                ingredients = technique.implement(ingredients = ingredients)
            self.depot.save(variable = ingredients.df)
        return ingredients

    def _implement_glob(self, ingredients):
        self.depot.initialize_writer(
                file_path = self.depot.path_out)
        ingredients.create_series()
        for file_num, a_path in enumerate(self.depot.path_in):
            if (file_num + 1) % 100 == 0 and self.verbose:
                print(file_num + 1, 'files parsed')
            with open(
                    a_path, mode = 'r', errors = 'ignore',
                    encoding = self.idea['files']['file_encoding']) as a_file:
                ingredients.source = a_file.implement()
                print(ingredients.df)
                ingredients.df[self.index_column] = file_num + 1
                for technique in self.techniques:
                    ingredients = technique.implement(ingredients = ingredients)
                self.depot.save(variable = ingredients.df)
        return ingredients

    def _set_columns(self, organizer):
        if not hasattr(self, 'columns'):
            self.columns = {self.index_column: int}
            if self.metadata_columns:
                self.columns.update(self.metadata_columns)
        self.columns.update(dict.fromkeys(self.columns, str))
        return self

    def draft(self) -> None:
        """ Declares default step names and classes in an Harvest."""
        super().draft()
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
        self._publish_techniques()
        self._publish_draft()
        if hasattr(self, '_set_folders'):
            self._set_folders()
        return self

    def publish(self, ingredients = None):
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
            self.ingredients = draft.implement(ingredients = self.ingredients)
            self.depot.save(variable = self.ingredients,
                                file_name = self.step + '_ingredients')
        return self