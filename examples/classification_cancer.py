
import os
import sys
sys.path.insert(0, os.path.join('..', 'simplify'))
sys.path.insert(0, os.path.join('..', '..', 'simplify'))

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from simplify import Idea

# Loads cancer data and converts from numpy arrays to a pandas DataFrame.
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns = np.append(cancer['feature_names'], ['target']))

# Sets root_folder for data and results exports.
root_folder = os.path.join('..', '..')
# Sets location of configuration settings for the project.
#idea_file = os.path.join(os.getcwd(), 'examples', 'cancer_settings.ini')
# Depending upon your OS and python configuration, this path might work better.
idea_file = os.path.join(os.getcwd(), 'cancer_settings.ini')

# Creates siMpLify project, automatically configuring the process based upon
# settings in the 'idea_file'.
cancer_project = Idea(
    configuration = idea_file,
    depot = root_folder,
    ingredients = df)

# Converts label to boolean type to correct numpy default above.
cancer_project.ingredients.change_datatype(columns = 'target',
                                           datatype = 'boolean')
# Fills missing data with appropriate default values based on column datatype.
cancer_project.ingredients.smart_fill()
# Iterates through every recipe and exports plots, explainers, and other
# metrics from each recipe.
cancer_project.implement()
# Outputs information about the best recipe to the terminal.
#cancer_project.critic.print_best()
# Saves ingredients file with predictions or predicted probabilities added
# (based on options in idea).
#cancer_project.ingredients.save(file_name = 'cancer_df')