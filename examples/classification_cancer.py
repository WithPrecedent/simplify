
import os
import sys
sys.path.insert(0, os.path.join('..', 'simplify'))

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from simplify import Idea, Depot, Ingredients
from simplify.chef import Cookbook

# Loads cancer data and converts from numpy arrays to pandas dataframe.
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns = np.append(cancer['feature_names'], ['target']))
# Initializes core simplify classes.
idea = Idea(configuration = os.path.join(os.getcwd(), 'examples',
                                         'cancer_settings.ini'))
depot = Depot(root_folder = os.path.join('..', '..'))
ingredients = Ingredients(df = df)
# Converts label to boolean type - conversion from numpy arrays leaves all
# columns as float type.
ingredients.change_datatype(columns = 'target', datatype = 'boolean')
# Fills missing ingredients with appropriate default values based on column
# datatype.
ingredients.smart_fill()
# Creates instance of Data with the cancer dataframe.
cookbook = Cookbook(ingredients = ingredients)
# Iterates through every recipe and exports plots, explainers, and other
# metrics from each recipe.
cookbook.produce()
# Saves the recipes, results, and cookbook.
cookbook.save_everything()
# Outputs information about the best recipe to the terminal.
cookbook.print_best()
# Saves ingredients file with predictions or predicted probabilities added
# (based on options in idea).
cookbook.ingredients.save(file_name = 'cancer_df')