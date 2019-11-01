"""
.. module:: chef composer
:synopsis: creates siMpLify-compatible algorithms for chef subpackage
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module

from simplify.chef.search_composer import SearchComposer
from simplify.core.compose import SimpleAlgorithm
from simplify.core.compose import SimpleComposer
from simplify.core.compose import SimpleTechnique
from simplify.core.decorators import numpy_shield


@dataclass
class ChefComposer(SimpleComposer):
    """[summary]

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
            
    """
    name: str = 'chef_composer'

    def __post_init__(self):
        super().__post_init__()
        # Declares parameter_types.
        self.parameter_types = (
            'idea',
            'selected',
            'extra',
            'search'
            'runtime',
            'conditional')
        return self

    """ Private Methods """

    def _get_search(self, technique: SimpleTechnique, parameters: dict):
        """[summary]

        Args:
            technique (SimpleTechnique): [description]
            parameters (dict): [description]

        """
        return parameters

    """ Core siMpLify Methods """

    def draft(self):
        """[summary]
        """
        # Subclasses should create Technique instances here.
        if self.gpu:
            self.add_gpu_techniques()
        return self

    def publish(self, technique: str, parameters: dict = None):
        """[summary]

        Args:
            technique (str): [description]
            parameters (dict, optional): [description]. Defaults to None.
        """
        if technique in ['none', 'None', None]:
            return None
        else:
            technique = getattr(self, '_'.join([step, technique]))
            algorithm = self._get_algorithm(technique = technique)
            parameters = self._get_parameters(
                technique = technique,
                parameters = parameters)
            return ChefAlgorithm(
                technique = technique.name,
                algorithm = algorithm,
                parameters = parameters,
                data_dependents = technique.data_dependents,
                hyperparameter_search = technique.hyperparameter_search,
                space = self.space)


@dataclass
class ChefTechnique(SimpleTechnique):

    name: str = 'chef_technique'
    module: str = None
    algorithm: str = None
    defaults: object = None
    extras: object = None
    runtimes: object = None
    data_dependents: object = None
    selected: bool = False
    conditional: bool = False
    hyperparameter_search: bool = False


@dataclass
class ChefAlgorithm(SimpleAlgorithm):
    """[summary]

    Args:
        object ([type]): [description]
    """
    technique: str
    parameters: object
    space: object

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Private Methods """


    def _datatype_in_list(self, test_list, datatype):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, datatype) for i in test_list)

    def _get_search(self, technique: ChefTechnique, parameters: dict):
        """[summary]

        Args:
            technique (ChefTechnique): [description]
            parameters (dict): [description]

        """
        self.space = {}
        if technique.hyperparameter_search:
            new_parameters = {}
            for parameter, values in parameters.items():
                if isinstance(values, list):
                    if self._datatype_in_list(values, float):
                        self.space.update(
                            {parameter: uniform(values[0], values[1])})
                    elif self._datatype_in_list(values, int):
                        self.space.update(
                            {parameter: randint(values[0], values[1])})
                else:
                    new_parameters.update({parameter: values})
            parameters = new_parameters
        return parameters

    def _search_hyperparameter(self, ingredients: Ingredients,
                               data_to_use: str):
        search = SearchComposer()
        search.space = self.space
        search.estimator = self.algorithm
        return search.publish(ingredients = ingredients)

    """ Core siMpLify Methods """

    @numpy_shield
    def publish(self, ingredients: Ingredients, data_to_use: str,
                columns: list = None, **kwargs):
        """[summary]

        Args:
            ingredients (Ingredients): [description]
            data_to_use (str): [description]
            columns (list, optional): [description]. Defaults to None.
        """
        if self.technique != 'none':
            if self.data_dependents:
                self._add_data_dependents(ingredients = ingredients)
            if self.hyperparameter_search:
                self.algorithm = self._search_hyperparameters(
                    ingredients = ingredients,
                    data_to_use = data_to_use)
            try:
                self.algorithm.fit(
                    X = getattr(ingredients, ''.join(['x_', data_to_use])),
                    Y = getattr(ingredients, ''.join(['y_', data_to_use])),
                    **kwargs)
                setattr(ingredients, ''.join(['x_', data_to_use]),
                        self.algorithm.transform(X = getattr(
                            ingredients, ''.join(['x_', data_to_use]))))
            except AttributeError:
                ingredients = self.algorithm.publish(
                    ingredients = ingredients,
                    data_to_use = data_to_use,
                    columns = columns,
                    **kwargs)
        return ingredients


    """ Scikit-Learn Compatibility Methods """

    def fit(self, x = None, y = None, ingredients = None):
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Raises:
            AttributeError if no 'fit' method exists for local 'algorithm'.

        """
        if hasattr(self.algorithm, 'fit'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    self.algorithm.fit(X = x)
                else:
                    self.algorithm.fit(X = x, Y = y)
            elif ingredients is not None:
                ingredients = self.algorithm.fit(
                    X = getattr(ingredients, 'x_' + self.data_to_train),
                    Y = getattr(ingredients, 'y_' + self.data_to_train))

        else:
            error = ('fit method does not exist for '
                     + self.technique + ' algorithm')
            raise AttributeError(error)
        return self

    def fit_transform(self, x = None, y = None, ingredients = None):
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Returns:
            transformed x or ingredients, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or ingredients is not passed to
                the method.

        """
        self.fit(x = x, y = y, ingredients = ingredients)
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            return self.transform(x = x, y = y)
        elif ingredients is not None:
            return self.transform(ingredients = ingredients)
        else:
            error = 'fit_transform requires DataFrame, ndarray, or Ingredients'
            raise TypeError(error)

    def transform(self, x = None, y = None, ingredients = None):
        """Generic transform method for partial compatibility to sklearn.

        Args:
            x(DataFrame or ndarray): independent variables/features.
            y(DataFrame, Series, or ndarray): dependent variable(s)/feature(s)
            ingredients(Ingredients): instance of Ingredients containing
                x_train and y_train attributes (based upon possible remapping).

        Returns:
            transformed x or ingredients, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'algorithm'.

        """
        if hasattr(self.algorithm, 'transform'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    return self.algorithm.transform(x)
                else:
                    return self.algorithm.transform(x, y)
            elif ingredients is not None:
                return self.algorithm.transform(
                    X = getattr(ingredients, 'x_' + self.data_to_train),
                    Y = getattr(ingredients, 'y_' + self.data_to_train))
        else:
            error = ('transform method does not exist for '
                     + self.technique + ' algorithm')
            raise AttributeError(error)
