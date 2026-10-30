import os
import numpy as np
import matplotlib.pyplot as plt
 
# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
N_EPOCHS = 1000          # number of passes over the training set
SEED = 0                 # fixed seed so the results are repeatable
STEP_SIZES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]   # for part (b)
SAVE_EVERY_EPOCH_FRAMES = False   # set True to save one fit plot per epoch (part c)
 
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
 
 
# ----------------------------------------------------------------------------
# Load data (each CSV row is "x,y," so only the first two columns are used)
# ----------------------------------------------------------------------------
def load(name):
    data = np.genfromtxt(os.path.join(HERE, name), delimiter=",", usecols=(0, 1))
    return data[:, 0], data[:, 1]
 
x_train, y_train = load("Dataset_2_train.csv")
x_valid, y_valid = load("Dataset_2_valid.csv")
x_test,  y_test  = load("Dataset_2_test.csv")
 
print(f"Train: {len(x_train)} points, Valid: {len(x_valid)}, Test: {len(x_test)}")

