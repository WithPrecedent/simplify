"""
harvest.py is the primary control file for the data gathering and processing
portions of the siMpLify package. It contains the Harvest class, which handles
the draftning and implementation for data gathering and preparation.
"""
from dataclasses import dataclass

from simplify.farmer.harvest import Harvest
from simplify.farmer.steps import Sow, Reap, Clean, Bundle, Deliver
from simplify.core.base import SimpleClass


@dataclass
class Almanac(SimpleClass):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Parameters:

        ingredients: an instance of Ingredients (or a subclass).
        steps: an ordered list of step names to be completed. This argument
            should only be passed if the user whiches to override the steps
            listed in idea.configuration.
        drafts: a list of instances of steps which Harvest creates
            through the finalize method and applies through the produce method.
            Ordinarily, a list of draft is not passed when Harvest is
            instanced, but the argument is included if the user wishes to
            reexamine past draft or manually add draft to an existing set.
            Alternatively, draft can be a dictionary of settings if the user
            prefers not to subclass Harvest and/or use .csv file imports,
            and instead pass the needed settings in dictionary form (with the
            keys corresponding to the names of techniques used and the values
            including the parameters to be used).
        name: a string designating the name of the class which should be
            identical to the section of the idea configuration with relevant
            settings.
        auto_finalize: a boolean value that sets whether the finalize method is
            automatically called when the class is instanced.
        auto_produce: sets whether to automatically call the 'produce' method
            when the class is instanced.

    """
    ingredients : object = None
    steps : object = None
    drafts : object = None
    name : str = 'harvest'
    auto_finalize : bool = True
    auto_produce : bool = True

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

    def draft(self):
        self.index_column = 'index_universal'
        self.metadata_columns = []
        return self

    def _finalize_draft(self):
        """Initializes the step classes for use by the Harvest."""
        self.drafts = []
        for step in self.steps:
            step_instance = self.draft_class(name = step,
                                            index_column = self.index_column)
            for technique in listify(getattr(self, step + '_techniques')):
                tool_instance = self.edit_technique(
                        step = step,
                        technique = technique,
                        parameters = listify(getattr(self, technique)))
                step_instance.techniques.append(tool_instance)
            step_instance.finalize()
            self.drafts.append(step_instance)
        return self

    def _set_columns(self, organizer):
        if not hasattr(self, 'columns'):
            self.columns = {self.index_column : int}
            if self.metadata_columns:
                self.columns.update(self.metadata_columns)
        self.columns.update(dict.fromkeys(self.columns, str))
        return self

    def draft(self):
        """ Declares default step names and classes in an Harvest."""
        super().draft()
        self.options = {'sow' : Sow,
                        'reap' : Reap,
                        'clean' : Clean,
                        'bundle' : Bundle,
                        'deliver' : Deliver}
        self.draft_class = Almanac
        self.checks.extend(['drafts', 'sections', 'defaults'])
        return self

    def finalize(self):
        """Creates a Harvest with all sequenced techniques applied at each
        step. Each set of methods is stored in a list within a Almanac instance.
        """
        if self.verbose:
            print('Preparing Harvest')
        self._finalize_draft_class()
        self._finalize_steps()
        self._finalize_draft()
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