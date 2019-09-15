
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain, product

from simplify.core.base import SimpleClass

    
@dataclass
class SimpleManager(SimpleClass, ABC):
      
    def __post_init__(self):
        super().__post_init__()
        # Outputs class status to console if verbose option is selected.
        if self.verbose:
            print('Creating', self)       
        return self

    """ Private Methods """
    
    def _finalize_parallel(self):
        """Creates all combinations of plans from user options and stores those 
        in 'plans' attribute.
        """
        self.plans = []
        step_combinations = []
        for step in self.options.keys():
            # Stores each step attribute in a list
            setattr(self, step, self.listify(getattr(self, step)))
            # Adds step to a list of all step lists
            step_combinations.append(getattr(self, step))
        # Creates a list of all possible permutations of step techniques
        # selected. Each item in the the list is an instance of the plan class.
        self.plans = list(map(list, product(*step_combinations)))
        return self
    
    def _finalize_serial(self):
        """Creates chained list of plans from user options and stores those 
        in 'plans' attribute.
        """
        self.plans = []
        step_chain = []
        for step in self.options.keys():
            # Stores each step attribute in a list
            setattr(self, step, self.listify(getattr(self, step)))
            # Adds step to a list of all step lists
            step_chain.append(getattr(self, step))
        # Creates a list of all step techniques selected in a chained list. Each 
        # item in the the list is an instance of the plan class.
        self.plans = list(map(list, chain(*step_chain)))          
        return self
        
    """ Public Methods """
    
    def draft(self):
        """ Declares defaults for class."""
        self.options = {}
        self.checks = ['steps', 'depot', 'ingredients']
        self.state_attributes = ['depot', 'ingredients']
        return self

    def finalize(self):
        getattr(self, '_finalize_', self.manager_type)
        return self
            
    @abstractmethod
    def produce(self, variable = None, **kwargs):
        """Required method that implements all of the finalized objects on the
        passed variable. The variable is returned after being transformed by
        called methods. It is roughly equivalent to the scikit-learn transform
        method.

        Parameters:
            variable: any variable. In most cases in the siMpLify package,
                variable is an instance of Ingredients. However, any variable
                or datatype can be used here.
            **kwargs: other parameters can be added to method as needed or
                **kwargs can be used.
        """
        pass
        return variable

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