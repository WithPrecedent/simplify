import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from simplify import Ingredients, Menu
from simplify.cookbook import Cookbook

# Loads cancer data and converts from numpy arrays to pandas dataframe.
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns = np.append(cancer['feature_names'], ['target']))
# Loads menu file.
menu = Menu(file_path = 'cancer_settings.ini')
# Creates instance of Data with the cancer ingredientsframe.
ingredients = Ingredients(menu = menu, df = df)
# Converts label to boolean type - conversion from numpy arrays leaves all
# columns as float type.
ingredients.change_type(columns = ['target'], datatype = bool)
# Fills missing ingredients with appropriate default values based on column type.
ingredients.smart_fillna()
# Creates instance of Cookbook.
cookbook = Cookbook(ingredients = ingredients, menu = menu)
# Automatically creates list of recipes cookbook.recipes based upon menu
# file.
cookbook.prepare()
# Iterates through every recipe and exports plots from each recipe.
cookbook.create()
# Creates and exports a table of summary statistics from the ingredientsframe.
ingredients.summarize()
# Saves the recipes, results, and cookbook.
cookbook.save_everything()
# Outputs information about the best recipe.
cookbook.print_best()
# Saves ingredients file.
ingredients.save(file_name = 'cancer_df')