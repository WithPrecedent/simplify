# siMpLify

![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
[![Build Status](https://img.shields.io/travis/with_precedent/siMpLify.svg)](https://travis-ci.org/with_precedent/siMpLify)

siMpLify offers tools to make data science more accessible, with a particular
emphasis on its use in academic research. To that end, the package avoids
programming jargon (when possible) and implements a unified code architecture
for all stages of a data science project. So, classes and methods for data
scraping, parsing, munging, merging, preprocessing, modelling, analyzing, and
visualizing use the same vocabulary so that siMpLify can be easily used and
extended.

siMpLify includes a high-level set of tools that allows users to mix and match various preprocessing methods and statistical models. It provides some unique custom methods and integrates classes and methods from packages such as scikit-learn, category-encoders, imblearn, xgboost, seaborn, and shap.

The siMpLify package uses an extended metaphor, which is familiar in computer
programming, as the basis for its overall structure: food preparation. Words
like 'recipe' and 'cookbook' appear with regularity in discussing computer code.
siMpLify extends this metaphor a bit further in the creation of its four core
packages:
    1) Wrangler: harvests data from a variety of sources, cleans it, and prepares
        it for consumption.
    2) Analyst: using a cookbook of recipes derived from user selections, the Analyst
        applies machine learning and preprocessing methods to data.
    3) Critic: evaluates the results of recipes, offering appropriately-matched       comparisons, summaries, and metrics.
    4) Artist: aiding the Critic, the Artist creates visualizations of the data,
        models, and model evaluation.

## Why siMpLify?

Although scikit-learn has gone a long way toward unifying interfaces with many common machine learning methods, it is still quite clunky in many situations. Present shortcomings include:
1) It doesn't incorporate many tools for data that isn't already [tidy](https://vita.had.co.nz/papers/tidy-data.pdf).
2) There is [a needlessly convoluted process](https://github.com/scikit-learn-contrib/sklearn-pandas#transformation-mapping) for implementing transformers on a subset of columns. Whereas many packages include a "cols" argument, [scikit-learn does not](https://medium.com/vickdata/easier-machine-learning-with-the-new-column-transformer-from-scikit-learn-c2268ea9564c).
3) fit methods do not work with certain preprocessing algorithms (e.g., [target encoding in category-encoders](https://github.com/scikit-learn-contrib/categorical-encoding/issues/104)) because scikit-learn does not allow the label data to be passed to a fit method.
4) Pipeline and FeatureUnion [lack a mix-and-match grid-search type system](https://buildmedia.implementthedocs.org/media/pdf/scikit-learn-enhancement-proposals/latest/scikit-learn-enhancement-proposals.pdf) for preprocessing, only for hyperparameter searches.
5) It doesn't directly use pandas dataframes despite various attempts to bridge the gap (e.g., [sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas)). This can cause confusion and difficulty in keeping feature names attached to columns of data because numpy arrays do not incorporate string names of columns. This is why, for example, [default feature_importances graphs do not include the actual feature names](https://stackoverflow.com/questions/44511636/matplotlib-plot-feature-importance-with-feature-names).
6) The structuring of scikit-learn compatible preprocessing algorithms to comply with the rigid .fit and .transform methods makes their use sometimes unintuitive.
7) The process for implementing different transformers on different groups of data (test, train, full, validation, etc.) within a Pipeline is [often messy and difficult](https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9).
8) Scikit-learn has [no plans to offer GPU support](https://scikit-learn.org/stable/faq.html#will-you-add-gpu-support).
9) Scikit-learn does not offer clear guidance to new users about how to sequence and combine [its many methods into a preprocessing and machine learning workflow](https://scikit-learn.org/stable/modules/classes.html).
10) Many great tools for machine learning, particularly in the category of "deep
learning" simply are not designed to be compatible with Scikit-learn.

siMpLify provides a cleaner, universal set of tools to access the many useful methods from scikit-learn and other python packages. The goal is to make machine learning more accessible to a wider user base. Simplify also adds numerous unique methods and functions for common machine learning and feature engineering workers. In addition to those custom scripts, siMpLify incorporates and provides a universal API for methods and classes from the following packages:

* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [xgboost](https://github.com/dmlc/xgboost)
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/imblearn)
* [categorical-encoding](https://github.com/scikit-learn-contrib/categorical-encoding)
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize/tree/master/skopt)
* [seaborn](https://github.com/mwaskom/seaborn)
* [shap](https://github.com/slundberg/shap)
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [eli5](https://github.com/TeamHG-Memex/eli5)
* [scikitplot](https://github.com/reiinakano/scikit-plot)

## The siMpLify Plan

To understand a typical use-case for siMplify, let's examine a project that omits the Wrangler stage and proceeds directly to preprocessing and modeling. At the end of this discussion of the general process, an example using the Wisconsin breast cancer data is included.

### siMpLify Analyst

As an example of siMpLify's functionality, let's review the Analyst subpackage. It allows users to create a cookbook of dynamic recipes that mix-and-match feature engineering and modeling dataset based upon a common, simple interface. It then analyzes the results using selected, appropriate metrics and exports tables, charts, and graphs compatible with the models and data types.

By default, the Analyst divides the feature engineering and modeling process into eight major steps that can be sequenced in different stepss (or supplemented with
custom steps and steps):

* Scale: converts numerical features into a common scale, using scikit-learn methods.
* Split: divides data into train, test, and/or validation sets once or iteratively through k-folds cross-validation.
* Encode: converts categorical features into numerical ones, using category-encoders methods.
* Mix: converts selected features into new polynomial features, using PolynomialEncoder from category-encoders or other mathmatical combinations.
* Cleave: creates different subgroups of features to allow for easy comparison between them. This stage is of particular importance to academic research and has
largely been omitted from existing efforts to simplify machine learning.
* Sample: synthetically resamples training data for imbalanced data, using imblearn methods, for use with models that struggle with imbalanced data.
* Reduce: selects features recursively or as one-shot based upon user criteria, using scikit-learn and prince methods.
* Model: implements machine learning algorithms. The user can opt to either test different hyperparameters for the models selected or a single set of hyperparameters. Hyperparameter earch methods currently include RandomizedSearchCV, GridSearchCV, and bayesian optimization through skopt.

### siMpLify Critic

As part of any machine learning workflow, assessment of prepared models is an essential entity. The Critic subpackage divides the evaluation process into four major stages:
* Summarize: building beyond the pandas describe method, this step includes a wide number of summary statistics for the user data, appropriately calculated based upon the data type of a particular variable.
* Score: automatically determining the compatibility of various scikit-learn and/or user-provided metrics, results for each recipe are calcuated.
* Evaluate: using explainers from shap, skater, and eli5, the various recipes are evaluated, feature importances calculated, and cumulative comparisons are made.
* Report: the above stages are compiled into appropriate reports which are exported to disk or, in some cases, outputted to the terminal.

### siMpLify Artist

Based upon the user selections and analysis done by the Critic, a set of visualizations is created for each recipe and as comparisons between recipes. Currently, this subpackage utilizes matplotlib, seaborn, shap, and a few other packages to make the visualization process easy using a common interface.

## siMpLify in Action - an Example

Perhaps the easiest, but not only, way to input user selections into the siMpLify package is by creating a simple text file (using the 'ini' format). This allows siMpLify to be used by beginner and advanced python programmers equally.

For example, using the settings file, a user could create a cookbook of recipes simply by listing the strings mapped to different methods:

    [cookbook]
    data_to_use = train_test
    model_type = classifier
    label = target
    calculate_hyperparameters = True
    naming_classes = model, cleaver
    export_all_recipes = True
    cookbook_steps = scaler, splitter, encoder, mixer, cleaver, sampler, reducer,   model
    scaler = normalizer, minmax
    splitter = train_test
    encoder = target
    mixer = polynomial
    cleaver = none
    sampler = smote, adasyn
    reducer = none
    model = xgboost, logit

With the above settings, all possible recipes are automatically created using either default or user-specified parameters. In total, there are eight recipes in the cookbook because two options are selected for the scaler, encoder, and model. Simply listing multiple choices separated by a comma is all that is needed for siMpLify to include and test different options.

siMpLify can also import hyperparameters from the text file, as illustrated below for the xgboost model:

    [xgboost]
    booster = gbtree
    objective = binary:logistic
    eval_metric = aucpr
    silent = True
    n_estimators = 50, 1000
    max_depth = 5, 15
    learning_rate = 0.001, 0.1
    subsample = 0.3
    colsample_bytree = 0.3
    colsample_bylevel = 0.3
    min_child_weight = 0.7, 1.0
    gamma = 0.0, 0.2
    alpha = 0.0, 0.2

In the above case, anywhere two values are listed separated by a comma, siMpLify automatically implements a hyperparameter search between those values (using the search method specified elsewhere in settings). If just one hyperparameter is listed, it stays fixed throughout the tests. Further, the hyperparameters are automatically linked to the 'xgboost' model by including that model name in the settings file. Further, if the 'gpu' setting is set to True (in the 'general' section of the settings file), the additional parameters needed to make xgboost use the local NVIDIA GPU will automatically be added.

The examples folder, from which the above settings are taken, currently shows how simplify works in analyzing the Wisconsin breast cancer database. The code for the analysis is relatively straightforward and simple:

    import os

    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer

    from simplify import Idea, inventory, Dataset
    from simplify.analyst import Cookbook

    # Loads cancer data and converts from numpy arrays to pandas dataframe.
    cancer = load_breast_cancer()
    df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                    columns = np.append(cancer['feature_names'], ['target']))
    # Initializes core simplify classes.
    idea = Idea(configuration = os.path.join(os.getcwd(), 'examples',
                                            'cancer_settings.ini'))
    inventory = inventory(root_folder = os.path.join('..', '..'))
    dataset = Dataset(df = df)
    # Converts label to boolean type - conversion from numpy arrays leaves all
    # columns as float type.
    dataset.change_datatype(columns = 'target', datatype = 'boolean')
    # Fills missing dataset with appropriate default values based on column
    # datatype.
    dataset.smart_fill()
    # Creates instance of Cookbook which, by default, will automatically create
    # all recipes from the settings file.
    cookbook = Cookbook(dataset = dataset)
    # Iterates through every recipe and exports plots, explainers, and other
    # metrics from each recipe.
    cookbook.implement()
    # Saves the recipes, results, and cookbook.
    cookbook.save_everything()
    # Outputs information about the best recipe to the terminal.
    cookbook.print_best()
    # Saves dataset file with predictions or predicted probabilities added
    # (based on options from the settings file).
    cookbook.dataset.save(file_name = 'cancer_df')

That's it. From that, all possible recipes are created. Each recipe gets its own folder within the results folder with relevant plots, a confusion matrix, and a classification report. A complete results file (review.csv) and summary statistics from the data (data_summary.csv) are stored in the results folder. Pickled cookbooks and recipes are also included if the user selects that option. In the above example, these are some of the plots automatically created for one of the recipes:

![](visuals/confusion_matrix.png.png?raw=true)
![](visuals/pr_curve.png.png?raw=true)
![](visuals/roc_curve.png.png?raw=true)
![](visuals/shap_heat_map.png.png?raw=true)
![](visuals/shap_summary.png.png?raw=true)
![](visuals/shap_interactions.png.png?raw=true)

New examples will be added showing different models and the Wrangler subpackage in the near future.