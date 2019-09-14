.. image:: visuals/siMpLify.png
   :width: 500px
   :align: center

siMpLify Documentation
======================
siMpLify allows users to create a cookbook of dynamic recipes that
mix-and-match feature engineering and modeling ingredients based upon a common,
simple interface. It then analyzes the results using selected, appropriate
metrics and exports tables, charts, and graphs compatible with the models and
data types.

siMpLify divides the feature engineering and modeling process into eleven
major steps that can be sequenced in different orders:

    Scaler: converts numerical features into a common scale, using scikit-learn
        methods.
    Splitter: divides data into train, test, and/or validation sets once or
        iteratively through k-folds cross-validation.
    Encoder: converts categorical features into numerical ones, using
        category-encoders methods.
    Interactor: converts selected categorical features into new polynomial
        features, using PolynomialEncoder from category-encoders or other
        mathmatical combinations.
    Splicer: creates different subgroups of features to allow for easy
        comparison between them.
    Sampler: synthetically resamples training data for imbalanced data,
        using imblearn methods, for use with models that struggle with
        imbalanced data.
    Selector: selects features recursively or as one-shot based upon user
        criteria, using scikit-learn methods.
    Custom: allows users to add any scikit-learn or siMpLify compatible
        method to be added into a recipe.
    Model: implements machine learning algorithms, currently includes
        xgboost and scikit-learn methods. The user can opt to either test
        different hyperparameters for the models selected or a single set
        of hyperparameters. Hyperparameter earch methods currently include
        RandomizedSearchCV, GridSearchCV, and bayesian optimization through
        skopt.
    Evaluator: tests the models using user-selected or default metrics and
        explainers from sklearn, shap, eli5, and lime.
    Plotter: produces helpful graphical representations based upon the model
        selected and evaluator and explainers used, utilizing seaborn and
        matplotlib methods.

Together, these steps form a recipe. Each recipe will be tested iteratively
using Cookbook.bake (or Cookbook.apply if preferred) or individually using
Recipe.bake (or Recipe.apply if preferred). If users choose to apply any of the
three steps in the recipe, results will be exported automatically.

siMpLify contains the following accessible classes:
    Cookbook: containing the methods needed to create dynamic recipes and
        stores them in Cookbook.recipes. For that reason, the Recipe class does
        not ordinarily need to be instanced directly.
    Recipe: if the user wants to manually create a single recipe, the Recipe
        class is made public for this purpose.
    Data: includes methods for creating and modifying pandas dataframes used
        by Cookbook. As the data is split into features, labels, test, train,
        etc. dataframes, they are all created as attributes to an instance of
        the Data class.
    Scaler, Splitter, Encoder, Interactor, Splicer, Sampler, Selector, and
        Custom: contain the different ingredient options for each step in a
        recipe.
    Model: contains different machine learning algorithms divided into three
        major model_type: classifier, regressor, and unsupervised.
    Results: contains the metrics used by Evaluator and stores a dataframe
        (Results.table) applying those metrics. Each row of the table stores
        each of the steps used, the folder in which the relevant files are
        stored, and all of the metrics used for that recipe.
    Evaluator: applies user-selected or default metrics for each recipe and
        passes those results for storage in Results.table.
    Plotter: finalizes and exports plots and other visualizations based upon
        the model type and Evaluator and Estimator methods.
    Filer: creates and contains the path structure for loading data and
        settings as well as saving results, data, and plots.
    Settings: contains the methods for parsing the settings file to create
        a nested dictionary used by the other classes.

If the user opts to use the settings.ini file, the only classes that absolutely
need to be used are Cookbook and Data. Nonetheless, the rest of the classes and
attributes are still available for use. All of the ten step classes are stored
in a list of recipes (Cookbook.recipes). Cookbook.evaluator collects the
results from the recipes being tested.

If a Filer instance is not passed to Cookbook when it instanced, an
import_folder and export_folder must be passed. Then the Cookbook will create
an instance of Filer as an attribute of the Cookbook (Cookbook.filer).
If an instance of Settings is not passed when the Cookbook is instanced, a
settings file will be loaded automatically.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Core Modules
======================

.. autoclass:: simplify.Cookbook
   :members:

.. autoclass:: simplify.Data
   :members:

.. autoclass:: simplify.Filer
   :members:

.. autoclass:: simplify.Recipe
   :members:

.. autoclass:: simplify.Results
   :members:

.. autoclass:: simplify.Settings
   :members:

Recipe Steps
======================

.. autoclass:: simplify.Scaler
   :members:

.. autoclass:: simplify.Splitter
   :members:

.. autoclass:: simplify.Encoder
   :members:

.. autoclass:: simplify.Interactor
   :members:

.. autoclass:: simplify.Sampler
   :members:

.. autoclass:: simplify.Selector
   :members:

.. autoclass:: simplify.Custom
   :members:

.. autoclass:: simplify.Model
   :members:

.. autoclass:: simplify.Evaluator
   :members:

.. autoclass:: simplify.Plotter
   :members:

siMpLify Utilities
======================

.. autofunction:: simplify.timer
   :members:

.. autoclass:: simplify.ReMatch
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
