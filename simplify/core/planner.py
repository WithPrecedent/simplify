
from dataclasses import dataclass

from simplify.core.base import SimpleClass


@dataclass
class Planner(SimpleClass):
    """Parent class for select siMpLify classes to provide shared methods for
    creating data science workflows. It can also be subclassed to create other
    Planners beyond those included in the package.
    """

    def __post_init__(self):
        super().__post_init__()
        # Outputs Planner status to console if verbose option is selected.
        if self.verbose:
            print('Creating', self)
        return self

    def draft(self):
        """ Declares defaults for Planner."""
        self.options = {}
        self.tools = ['listify']
        self.draft_class = None
        self.checks = ['steps', 'depot', 'ingredients']
        self.state_attributes = ['depot', 'ingredients']
        return self

    # def edit_options(self, step = None, techniques = None, algorithms = None):
    #     """Adds new technique name and corresponding algorithm to the options
    #     dictionary.
    #     """
    #     if step:
    #         self.options[step].edit_options(techniques = techniques,
    #                                        algorithms = algorithms)
    #     else:
    #         options = dict(zip(self.listify(techniques),
    #                            self.listify(algorithms)))
    #         self.options.update(options)
    #     return self

    # def edit_parameters(self, step, parameters):
    #     """Adds parameter sets to the parameters dictionary of a prescribed
    #     step. """
    #     self.options[step].edit_parameters(parameters = parameters)
    #     return self

    # def edit_runtime_parameters(self, step, parameters):
    #     """Adds runtime_parameter sets to the parameters dictionary of a
    #     prescribed step."""
    #     self.options[step].edit_runtime_parameters(parameters = parameters)
    #     return self

    # def edit_step_class(self, step_name, step_class):
    #     self.options.update({step_name, step_class})
    #     return self

    # def edit_technique(self, step, technique, parameters = None):
    #     tool_instance = self.options[step](technique = technique,
    #                                        parameters = parameters)
    #     return tool_instance

    def conform(self, step = None):
        if not step:
            step = self.step
        for attribute in self.state_attributes:
            getattr(self, attribute).conform(step = step)
        return self

    def save(self):
        """Exports the list of drafts to disc as one object."""
        self.depot.save(variable = self,
                            folder = self.depot.experiment,
                            file_name = self.name,
                            file_format = 'pickle')
        return self

    def save_draft(self, draft, file_path = None):
        """Saves an instance of the draft class."""
        self.depot.save(variable = draft,
                            file_path = file_path,
                            folder = getattr(self.depot, draft.name),
                            file_name = 'recipe',
                            file_format = 'pickle')
        return self