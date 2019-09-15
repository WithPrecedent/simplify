"""
harvest.py is the primary control file for the data gathering and processing
portions of the siMpLify package. 

Contents:
    Almanac: class which handles construction and utilization of harvests of
        data gathering, parsing, munging, merging, and staging of data in
        the siMpLify package.
    Harvest: class which stores a particular set of techniques and algorithms
        for data gathering, parsing, munging, merging, and staging of data in
        the siMpLify package.
        
    Both classes are subclasses to SimpleClass and follow its structural rules.
"""
from dataclasses import dataclass

from simplify.farmer.harvest import Harvest
from simplify.farmer.steps import Sow, Reap, Clean, Bale, Deliver
from simplify.core.base import SimpleClass


@dataclass
class Almanac(SimpleClass):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Parameters:

        ingredients: an instance of Ingredients (or a subclass). This argument
            need not be passed when the class is instanced, particularly if
            the user in the early stages of data gathering and initial parsing.
        steps: a list of string step names to be completed in order. This 
            argument should only be passed if the user wishes to override the 
            steps listed in the Idea settings or if the user is not using the
            Idea class.
        harvests: a list of instances of Harvest. Ordinarily, a list of draft is 
            not passed when Harvest is instanced, but the argument is included 
            if the user wishes to use previously built harvest techniques.
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
    harvests : object = None
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
                        'bale' : Bale,
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
    

@dataclass
class Harvest(object):
    """Defines rules for sowing, reaping, cleaning, bundling, and delivering
    data as part of the siMpLify Harvest subpackage.

    Attributes:
        techniques: a list of Harvest step techniques to complete.
        name: a string designating the name of the class which should be
            identical to the section of the idea with relevant settings.
    """
    techniques : object = None
    name : str = 'draft'
    index_column : str = 'index_universal'

    def __post_init__(self):
        if not self.techniques:
            self.techniques = []
        return self
#
#    def _check_attributes(self):
#        for technique in self.techniques:
#            if not hasattr(self, technique.technique):
#                error = technique.technique + ' has not been passed to class.'
#                raise AttributeError(error)
#        return self
#
#    def _set_columns(self, variable):
#        if (not hasattr(self, 'column_names')
#            and variable.technique in ['organize']):
#            self.column_names = listify(self.index_column)
#            self.column_names.extend(variable.columns)
#        elif not hasattr(self, 'column_names'):
#            self.column_names = list(variable.columns.keys())
#        return self

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

    def finalize(self):
#        self._check_attributes()
#        for technique in self.techniques:
#            self._set_columns(variable = technique)
        return self

    def produce(self, ingredients):
        """Applies the Harvest technique classes to the passed ingredients."""
        if isinstance(self.depot.path_in, list):
            ingredients = self._produce_glob(ingredients = ingredients)
        else:
            ingredients = self._produce_file(ingredients = ingredients)
        return ingredients