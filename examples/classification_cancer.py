import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from cookbook import Cookbook
from data import Data
from settings import Settings

# Loads cancer data and convert from numpy arrays to pandas dataframe.
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns = np.append(cancer['feature_names'], ['target']))
# Loads settings file.
settings = Settings(file_path = 'cancer_settings.ini')
# Create instance of data with the cancer dataframe.
data = Data(settings = settings, df = df)
# Converts label to boolean type - conversion from numpy arrays leaves all
# columns as float type.
data.change_column_type(columns = ['target'], data_type = bool)
# Fills missing data with appropriate default values based on column type.
data.smart_fillna()
# Creates instance of Cookbook.
cookbook = Cookbook(data = data, settings = settings)
# Automatically creates list of recipes cookbook.recipes based upon settings
# file.
cookbook.create()
# Iterates through every recipe and exports results and plots from each recipe.
cookbook.iterate()
# Saves the recipes, results, and cookbook.
cookbook.save_everything()
# Outputs information about the best recipe.
cookbook.print_best()
# Saves data file.
data.save(file_name = 'cancer_df')