
import os
import sys
sys.path.insert(0, os.path.join('..', 'simplify'))

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from simplify import Simplify, Idea

# Sets path to settings file and root folder for file output.
idea = Idea(configuration = os.path.join(os.getcwd(), 'examples', 
                                         'cancer_settings.ini'))
root_folder = os.path.join('..', '..')

# Loads cancer data and converts from numpy arrays to pandas dataframe.
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns = np.append(cancer['feature_names'], ['target']))

# Creates simplify instance to process and analyze data.
cancer_project = Simplify(ingredients = df,
                          idea = idea,
                          depot = root_folder)
# Converts label to boolean type to correct numpy default above.
cancer_project.ingredients.change_datatype(columns = 'target', 
                                           datatype = 'boolean')
# Fills missing data with appropriate default values based on column datatype.
cancer_project.ingredients.smart_fill()
# Iterates through every recipe and exports plots, explainers, and other
# metrics from each recipe.
cancer_project.produce()
# Saves the recipes, results, and cookbook.
cancer_project.chef.save_everything()
# Outputs information about the best recipe to the terminal.
cancer_project.critic.print_best()
# Saves ingredients file with predictions or predicted probabilities added
# (based on options in idea).
cancer_project.ingredients.save(file_name = 'cancer_df')