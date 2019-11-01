"""
.. module:: validator
:synopsis: checks key attributes and loads/converts them as needed.
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.depot import Depot
from simplify.core.idea import Idea
from simplify.core.ingredients import Ingredients


@dataclass
class Validator(object):
    """Performs attribute validation and type conversion as neceessary.

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.

    """
    name: str = 'validator'

    def __post_init__(self):
        self.draft()
        return self

    """ Private Methods """

    def _check_depot(self, instance = None):
        """Adds a Depot instance with default settings as 'depot' attribute if
        one was not passed when 'instance' was instanced.

        Args:
            instance (SimpleClass): subclass to be checked.

        Raises:
            TypeError: if 'depot' is neither a str, None, nor Depot instance.

        """
        if instance.exists('depot'):
            if isinstance(instance.depot, str):
                instance.depot = Depot(root_folder = instance.depot)
            elif isinstance(instance.depot, Depot):
                pass
            else:
                error = 'depot must be a string type or Depot instance'
                raise TypeError(error)
        else:
            instance.depot = Depot()
        self._inject_base(attribute = 'depot')
        return instance

    def _check_gpu(self, instance = None):
        """If gpu status is not set, checks if the local machine has a GPU
        capable of supporting included machine learning algorithms.

        Because the tensorflow 'is_gpu_available' method is very lenient in
        counting what qualifies, it is recommended to set the 'gpu' attribute
        directly or through an Idea instance.

        Args:
            instance (SimpleClass): subclass to be checked.

        """
        if hasattr(self, 'gpu'):
            if instance.gpu and instance.verbose:
                print('Using GPU')
            elif instance.verbose:
                print('Using CPU')
        else:
            try:
                from tensorflow.test import is_gpu_available
                if is_gpu_available:
                    instance.gpu = True
                    if instance.verbose:
                        print('Using GPU')
            except ImportError:
                instance.gpu = False
                if instance.verbose:
                    print('Using CPU')
        return instance

    def _check_idea(self, instance = None, idea = None):
        """Checks if 'ingredients' attribute exists and takes appropriate
        action.

        If an 'ingredients' attribute exists, it determines if it contains a
        file folder, file path, or Ingredients instance. Depending upon its
        type, different actions are taken to actually create an Ingredients
        instance.

        If ingredients is None, then an Ingredients instance is created with no
        pandas DataFrames or Series within it.

        Args:
            instance (SimpleClass): subclass to be checked.
            ingredients (Ingredients, a file path containing a DataFrame or
                Series to add to an Ingredients instance, a folder
                containing files to be used to compose Ingredients DataFrames
                and/or Series, DataFrame, Series, or numpy array).

        Raises:
            TypeError: if 'ingredients' is neither a str, None, DataFrame,
                Series, numpy array, or Ingredients instance.

        """
        if idea is None:
            idea = instance.idea

        return instance


    def _check_ingredients(self, instance = None, ingredients = None):
        """Checks if 'ingredients' attribute exists and takes appropriate
        action.

        If an 'ingredients' attribute exists, it determines if it contains a
        file folder, file path, or Ingredients instance. Depending upon its
        type, different actions are taken to actually create an Ingredients
        instance.

        If ingredients is None, then an Ingredients instance is created with no
        pandas DataFrames or Series within it.

        Args:
            instance (SimpleClass): subclass to be checked.
            ingredients (Ingredients, a file path containing a DataFrame or
                Series to add to an Ingredients instance, a folder
                containing files to be used to compose Ingredients DataFrames
                and/or Series, DataFrame, Series, or numpy array).

        Raises:
            TypeError: if 'ingredients' is neither a str, None, DataFrame,
                Series, numpy array, or Ingredients instance.

        """

        if ingredients is None:
            instance.ingredients = Ingredients()
        else:
            instance.ingredients = ingredients
            # If 'ingredients' is a data container, it is assigned to 'df' in a
            # new instance of Ingredients assigned to 'ingredients'.
            # If 'ingredients' is a file path, the file is loaded into a
            # DataFrame and assigned to 'df' in a new Ingredients instance at
            # 'ingredients'.
            # If 'ingredients' is None, a new Ingredients instance is created
            # and assigned to 'ingreidents' with no attached DataFrames.
            if (isinstance(instance.ingredients, pd.Series)
                    or isinstance(instance.ingredients, pd.DataFrame)
                    or isinstance(instance.ingredients, np.ndarray)):
                instance.ingredients = Ingredients(df = instance.ingredients)
            elif isinstance(instance.ingredients, str):
                if os.path.isfile(instance.ingredients):
                    df = instance.depot.load(
                        folder = instance.depot.data,
                        file_name = instance.ingredients)
                    instance.ingredients = Ingredients(df = df)
                elif os.path.isdir(instance.ingredients):
                    instance.depot.create_glob(folder = instance.ingredients)
                    instance.ingredients = Ingredients()
            else:
                error = ' '.join('ingredients must be a str, DataFrame, None,',
                                 'numpy array, or, an Ingredients instance')
                raise TypeError(error)
        return instance

    def _check_name(self, instance = None):
        """Sets 'name' attribute if one does not exist in subclass.

        A separate 'name' attribute is used throughout the package so that
        users can set their own naming conventions or use the names of parent
        classes when subclassing without being dependent upon
        __class__.__name__.

        If no 'name' attribute exists, then __class__.__name__.lower() is used
        as the default 'name'.

        """
        if not instance.exists('name'):
            instance.name = instance.__class__.__name__.lower()
        return instance

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        return self

    def publish(self, instance = None, checks = None):
        """Checks attributes from 'checks' and runs corresponding methods based
        upon strings stored in 'checks'.

        Those methods should have the prefix '_check_' followed by the string
        in the attribute 'checks'. Any subclass seeking to add new checks can
        add a new method using those naming conventions with the only passed
        argument as 'self'.

        Args:
            instance(SimpleClass): instance to have checks run against.
            checks(list or str): list or name of checks to be run. If not
                passed, the list in the 'checks' attribute with be used.

        Returns:
            instance(SimpleClass): instance will modifications made based upon
                checks performed.
        Raises:
            AttributeError: if correspondig method name is neither in the
                passed instance nor this class instance.
        """
        print(instance.name, checks)
        if not checks and hasattr(instance, 'checks'):
            checks = instance.checks
        if checks:
            for check in self.listify(checks):
                try:
                    getattr(instance, '_check_' + check)()
                except (TypeError, AttributeError):
                    try:
                        print('looking in validator', instance.name, check)
                        print(getattr(self, '_check_' + check))
                        instance = getattr(self, '_check_' + check)(
                            instance = instance)
                    except (TypeError, AttributeError):
                        error = check + ' does not correspond to method name'
                        raise AttributeError(error)
        return instance