
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



    def _define(self):
        """ Declares defaults for Planner."""
        self.options = {}
        self.tools = ['listify']
        self.plan_class = None
        self.checks = ['steps', 'inventory', 'ingredients']
        self.state_attributes = ['inventory', 'ingredients']
        return self

    # def add_options(self, step = None, techniques = None, algorithms = None):
    #     """Adds new technique name and corresponding algorithm to the options
    #     dictionary.
    #     """
    #     if step:
    #         self.options[step].add_options(techniques = techniques,
    #                                        algorithms = algorithms)
    #     else:
    #         options = dict(zip(self.listify(techniques),
    #                            self.listify(algorithms)))
    #         self.options.update(options)
    #     return self

    # def add_parameters(self, step, parameters):
    #     """Adds parameter sets to the parameters dictionary of a prescribed
    #     step. """
    #     self.options[step].add_parameters(parameters = parameters)
    #     return self

    # def add_runtime_parameters(self, step, parameters):
    #     """Adds runtime_parameter sets to the parameters dictionary of a
    #     prescribed step."""
    #     self.options[step].add_runtime_parameters(parameters = parameters)
    #     return self

    # def add_step_class(self, step_name, step_class):
    #     self.options.update({step_name, step_class})
    #     return self

    # def add_technique(self, step, technique, parameters = None):
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
        """Exports the list of plans to disc as one object."""
        self.inventory.save(variable = self,
                            folder = self.inventory.experiment,
                            file_name = self.name,
                            file_format = 'pickle')
        return self

    def save_plan(self, plan, file_path = None):
        """Saves an instance of the plan class."""
        self.inventory.save(variable = plan,
                            file_path = file_path,
                            folder = getattr(self.inventory, plan.name),
                            file_name = 'recipe',
                            file_format = 'pickle')
        return self