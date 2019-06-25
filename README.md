# siMpLify

![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
[![Build Status](https://img.shields.io/travis/with_precedent/siMpLify.svg)](https://travis-ci.org/with_precedent/siMpLify)

![siMpLify](visuals/siMpLify.png?raw=true "siMpLify")

siMpLify is a high-level set of tools that allows users to mix and match various preprocessing methods and statistical models. It provides some unique custom methods and integrates classes and methods from packages such as scikit-learn, category-encoders, imblearn, xgboost, seaborn, and shap. Users are then able to test and implement different recipes in a machine learning cookbook and export results for each combination of options selected.

Although scikit-learn has gone a long way toward unifying interfaces with many common machine learning methods, it is still quite clunky in many situations. Present shortcomings include:
1) There is [a needlessly convoluted process](https://github.com/scikit-learn-contrib/sklearn-pandas#transformation-mapping) for implementing transformers on a subset of columns. Whereas many packages include a "cols" argument, [scikit-learn does not](https://medium.com/vickdata/easier-machine-learning-with-the-new-column-transformer-from-scikit-learn-c2268ea9564c).
2) fit methods do not work with certain preprocessing algorithms (e.g., [target encoding in category-encoders](https://github.com/scikit-learn-contrib/categorical-encoding/issues/104)) because scikit-learn does not allow the label data to be passed to a fit method.
3) Pipeline and FeatureUnion [lack a mix-and-match type grid-search system](https://buildmedia.readthedocs.org/media/pdf/scikit-learn-enhancement-proposals/latest/scikit-learn-enhancement-proposals.pdf) for preprocessing, only for hyperparameter searches.
4) It doesn't directly use pandas dataframes despite various attempts to bridge the gap (e.g., [sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas)). This can cause confusion and difficulty in keeping feature names attached to columns of data because numpy arrays do not incorporate string names of columns. This is why, for example, [default feature_importances graphs do not include the actual feature names](https://stackoverflow.com/questions/44511636/matplotlib-plot-feature-importance-with-feature-names).
5) The structuring of scikit-learn compatible preprocessing algorithms to comply with the rigid .fit and .transform methods makes their use sometimes unintuitive.
6) The process for implementing different transformers on different groups of data (test, train, full, validation, etc.) within a Pipeline is [often messy and difficult](https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9).
7) Scikit-learn has [no plans to offer GPU support](https://scikit-learn.org/stable/faq.html#will-you-add-gpu-support).
8) Scikit-learn does not offer clear guidance to new users about how to sequence and combine [its many methods into a preprocessing and machine learning workflow](https://scikit-learn.org/stable/modules/classes.html).
9) Many great tools for machine learning simply are not designed to be compatible with Scikit-learn.

siMpLify provides a cleaner, universal set of tools to access the many useful methods from scikit-learn and other python open-source packages. The goal is to make machine learning more accessible to a wider user base. Simplify also adds numerous unique methods and functions for common machine learning and feature engineering tasks. In addition to those custom scripts, siMpLify incorporates and provides a universal API for methods and classes from the following packages:

* [sklearn](https://github.com/scikit-learn/scikit-learn)
* [xgboost](https://github.com/dmlc/xgboost)
* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/imblearn)
* [categorical-encoding](https://github.com/scikit-learn-contrib/categorical-encoding)
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize/tree/master/skopt)
* [seaborn](https://github.com/mwaskom/seaborn)
* [shap](https://github.com/slundberg/shap)
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [eli5](https://github.com/TeamHG-Memex/eli5)
* [scikitplot](https://github.com/reiinakano/scikit-plot)

The siMpLify package allows users to create a cookbook of dynamic recipes that mix-and-match feature engineering and modeling ingredients based upon a common, simple interface. It then analyzes the results using selected, appropriate metrics and exports tables, charts, and graphs compatible with the models and data types.

siMpLify divides the feature engineering and modeling process into eight major steps that can be sequenced in different orders:

* Scale: converts numerical features into a common scale, using scikit-learn methods.
* Split: divides data into train, test, and/or validation sets once or iteratively through k-folds cross-validation.
* Encode: converts categorical features into numerical ones, using category-encoders methods.
* Mix: converts selected features into new polynomial features, using PolynomialEncoder from category-encoders or other mathmatical combinations.
* Cleave: creates different subgroups of features to allow for easy comparison between them.
* Sample: synthetically resamples training data for imbalanced data, using imblearn methods, for use with models that struggle with imbalanced data.
* Reduce: selects features recursively or as one-shot based upon user criteria, using scikit-learn methods.
* Model: implements machine learning algorithms, currently includes xgboost and scikit-learn methods. The user can opt to either test different hyperparameters for the models selected or a single set of hyperparameters. Hyperparameter earch methods currently include RandomizedSearchCV, GridSearchCV, and bayesian optimization through skopt.

Users can also easily add other steps to the process and use the Cookbook to incorporate those steps into the tested and evaluated recipes. In addition to the algorithms included, users can easily add additional algorithms at any inclued step by calling easy-to-use class methods. Users can easily select to use some or all of the above steps in specific cases with one or more methods selected at each stage. The order is also largely dynamic so that steps can be rearranged based upon user choice.

In addition to the steps for cookbook recipes, siMpLify includes a critic subpackage which assesses the recipe and produces appropriate visualizations. The two primary classes of the critic are:
* Review: tests the recipes using user-selected or default metrics and explainers from sklearn, shap, and eli5. A report attribute is created and exported which contains the results for all recipes in a cookbook (or recipes passed directly to a Review instance).
* Presentation: produces helpful graphical representations based upon the model selected and evaluator and explainers used, utilizing seaborn, shap, and matplotlib methods.


Users can easily select different options using a text settings file (siMpLify_settings.ini) or passing appropriate dictionaries to Cookbook. This allows siMpLify to be used by beginner and advanced python programmers equally.

For example, using the settings file, a user could create a cookbook of recipes simply by listing the strings mapped to different methods:

    [recipes]
    order = scaler, splitter, encoder, mixer, sampler, reducer, model
    scaler = minmax, normalizer
    splitter = train_test
    encoder = target, helmert
    mixer = polynomial
    cleaver = none
    sampler = smote
    reducer = none
    model = xgb, logit
    metrics = roc_auc, f1, accuracy, balanced_accuracy, brier_score_loss, hamming, jaccard, neg_log_loss, matthews_corrcoef, precision, recall, zero_one
    search_algorithm = random
    export_all_recipes = True

With the above settings, all possible recipes are automatically created using either default or user-specified parameters. In total, there are eight recipes in the cookbook because 2 options are selected for the scaler, encoder, and model.

siMpLify can import hyperparameters (and/or hyperparameter searches) based upon user selections in the settings file as follows:

    [xgb_params]
    booster = gbtree
    objective = binary:logistic
    eval_metric = aucpr
    silent = True
    n_estimators = 50, 500
    max_depth = 3, 30
    learning_rate = 0.001, 1.0
    subsample = 0.2, 0.8
    colsample_bytree = 0.2, 0.8
    colsample_bylevel = 0.2, 0.8
    min_child_weight = 0.5, 1.5
    gamma = 0.0, 0.1
    alpha = 0.0, 1

In the above case, anywhere two values are listed separated by a comma, siMpLify automatically implements a hyperparameter search between those values (using either randint or uniform from scipy.stats, depending upon the data type). If just one hyperparameter is listed, it stays fixed throughout the tests. Further, the hyperparameters are automatically linked to the 'xgb' model by including that prefix to '_params' in the settings file.

siMpLify currently supports NVIDIA GPU modeling for xgboost and will implement it for other incorporated models with built-in GPU support.

In addition to the implementation of the steps above, summary statistics and results (appropriate to the models) are exported to user-selected formats in either dynamically created or selected paths.

The examples folder, from which the above settings are taken, currently shows how simplify works in analyzing the Wisconsin breast cancer database. The code for the analysis is relatively straightforward and simple:

    import pandas as pd
    import numpy as np

    from sklearn.datasets import load_breast_cancer

    from simplify import Ingredients, Inventory, Menu
    from simplify.cookbook import Cookbook

    # Loads cancer data and converts from numpy arrays to pandas dataframe.
    cancer = load_breast_cancer()
    df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                      columns = np.append(cancer['feature_names'], ['target']))
    # Loads menu file.
    menu = Menu(file_path = 'cancer_settings.ini')
    inventory = Inventory(menu = menu)
    # Creates instance of Data with the cancer dataframe.
    ingredients = Ingredients(df = df, menu = menu, inventory = inventory)
    # Converts label to boolean type - conversion from numpy arrays leaves all
    # columns as float type.
    ingredients.change_type(columns = ['target'], datatype = bool)
    # Fills missing ingredients with appropriate default values based on column type.
    ingredients.smart_fillna()
    # Creates instance of Cookbook.
    cookbook = Cookbook(ingredients = ingredients, menu = menu,
                        inventory = inventory)
    # Automatically creates list of recipes cookbook.recipes based upon menu
    # file.
    cookbook.prepare()
    # Iterates through every recipe and exports plots from each recipe.
    cookbook.create()
    # Creates and exports a table of summary statistics from the dataframe.
    ingredients.summarize()
    # Saves the recipes, results, and cookbook.
    cookbook.save_everything()
    # Outputs information about the best recipe.
    cookbook.print_best()
    # Saves ingredients file.
    ingredients.save(file_name = 'cancer_df')

That's it. From that, all possible recipes are created. Each recipe gets its own folder within the results folder with relevant plots, a confusion matrix, and a classification report. A complete results file (review.csv) and summary statistics from the data (data_summary.csv) are stored in the results folder. Pickled cookbooks and recipes are also included if the user selects that option. In the above example, these are some of the plots automatically created for one of the recipes:

![](visuals/confusion_matrix.png?raw=true)
![](visuals/pr_curve.png?raw=true)
![](visuals/roc_curve.png?raw=true)
![](visuals/shap_heat_map.png?raw=true)
![](visuals/shap_summary.png?raw=true)
![](visuals/shap_interactions.png?raw=true)

Documentation and the Almanac class, which aids with data munging, wrangling, and parsing, are forthcoming.

