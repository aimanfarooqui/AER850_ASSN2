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

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def design_matrix(x):
    """Add a column of ones so w[0] acts as the intercept."""
    return np.column_stack([np.ones_like(x), x])
 
def mse(w, x, y):
    return np.mean((design_matrix(x) @ w - y) ** 2)
 
def train_sgd(step_size, n_epochs=N_EPOCHS, seed=SEED):
    """
    Train y_hat = w0 + w1*x with SGD.
    For one sample (x_i, y_i) the squared error is e = (w.x_i - y_i)^2,
    and its gradient is  de/dw = 2 * (w.x_i - y_i) * x_i.
    Returns the final weights, the weights after every epoch, and the
    training / validation MSE after every epoch.
    """
    rng = np.random.default_rng(seed)
    X = design_matrix(x_train)
    w = np.zeros(2)                       # start from w0 = w1 = 0
    w_hist, train_hist, valid_hist = [], [], []
 
    with np.errstate(over="ignore", invalid="ignore"):
        for epoch in range(n_epochs):
            for i in rng.permutation(len(x_train)):     # shuffle each epoch
                error = X[i] @ w - y_train[i]
                w = w - step_size * 2 * error * X[i]
            w_hist.append(w.copy())
            train_hist.append(mse(w, x_train, y_train))
            valid_hist.append(mse(w, x_valid, y_valid))
            if not np.isfinite(train_hist[-1]):         # diverged, stop early
                break
 
    return w, np.array(w_hist), np.array(train_hist), np.array(valid_hist)