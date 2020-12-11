"""
.. module:: manual
:synopsis: data gathering, munging, and preprocessing content module
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass

from simplify.core.utilities import local_backups
from simplify.core.package import SimpleBook
from simplify.core.definitionsetter import WranglerTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleDirector
subclass because siMpLify uses a lazy importing core. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleDirector subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'sow': ['simplify.wrangler.sow', 'Sow'],
    'harvest': ['simplify.wrangler.harvest', 'Harvest'],
    'clean': ['simplify.wrangler.clean', 'Clean'],
    'bale': ['simplify.wrangler.bale', 'Bale'],
    'deliver': ['simplify.wrangler.deliver', 'Deliver']}


@dataclasses.dataclass
class Manual(SimpleBook):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Args:
        idea(Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located.
        filer(filer): an instance of filer.
        dataset(Dataset or str): an instance of Dataset or a string
            with the file path for a pandas DataFrame that will. This argument
            does not need to be passed when the class is instanced.
        steps(dict(str: WranglerTechnique)): steps to be completed in steps. This
            argument should only be passed if the user wishes to override the
            steps listed in the Idea settings or if the user is not using the
            Idea class.
        plans(SimpleBook): instanced subclasses of SimpleBook for
            prepared tools for the Manual.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_draft(bool): whether to call the 'publish' method when the
            class is instanced. If you do not plan to make any
            adjustments to the steps, steps, or algorithms beyond the
            Idea configuration, this option should be set to True. If you plan
            to make such changes, 'publish' should be called when those
            changes are complete.
        auto_publish(bool): whether to call the 'publish' method when the
            class is instanced.

    Since this class is a subclass to SimpleBook and SimpleDirector, all
    documentation for those classes applies as well.

    """
    idea: object = None
    filer: object = None
    dataset: object = None
    steps: object = None
    plans: object = None
    name: str = 'analyst'
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
        for step in self.steps:
            step_instance = self.draft_class(name = step,
                                            index_column = self.index_column)
            for step in utilities.listify(getattr(self, step + '_steps')):
                tool_instance = self.edit_step(
                        step = step,
                        step = step,
                        parameters = utilities.listify(getattr(self, step)))
                step_instance.steps.append(tool_instance)
            step_instance.publish()
            self.drafts.append(step_instance)
        return self

    def _implement_file(self, dataset):
        with open(
                self.filer.path_in, mode = 'r', errors = 'ignore',
                encoding = self.workers.idea['files']['file_encoding']) as a_file:
            dataset.source = a_file.implement()
            for step in self.steps:
                data = step.implement(data = dataset)
            self.filer.save(variable = dataset.df)
        return dataset

    def _implement_glob(self, dataset):
        self.filer.initialize_writer(
                file_path = self.filer.path_out)
        dataset.make_series()
        for file_num, a_path in enumerate(self.filer.path_in):
            if (file_num + 1) % 100 == 0 and self.verbose:
                print(file_num + 1, 'files parsed')
            with open(
                    a_path, mode = 'r', errors = 'ignore',
                    encoding = self.workers.idea['files']['file_encoding']) as a_file:
                dataset.source = a_file.implement()
                print(dataset.df)
                dataset.df[self.index_column] = file_num + 1
                for step in self.steps:
                    data = step.implement(data = dataset)
                self.filer.save(variable = dataset.df)
        return dataset

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
        self.draft_class = Manual
        self.checks.extend(['drafts', 'sections', 'defaults'])
        return self

    def publish(self):
        """Creates a Harvest with all sequenced steps applied at each
        step. Each set of methods is stored in a list within a Manual instance.
        """
        if self.verbose:
            print('Preparing Harvest')
        self._publish_draft_class()
        self._publish_steps()
        self._publish_draft()
        if hasattr(self, '_set_folders'):
            self._set_folders()
        return self

    def publish(self, data = None):
        """Completes an iteration of an Harvest."""
        if not dataset:
            data = self.dataset
        for draft in self.drafts:
            self.step = draft.name
            # Adds initial columns dictionary to dataset instance.
            if (self.step in ['reap']
                    and 'organize' in self.reap_steps):
                self._set_columns(organizer = draft)
                dataset.columns = self.columns
            self.conform(step = self.step)
            self.data = draft.implement(data = self.dataset)
            self.filer.save(variable = self.dataset,
                                file_name = self.step + '_dataset')
        return self