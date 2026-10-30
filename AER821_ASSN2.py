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

# ----------------------------------------------------------------------------
# Part (a): step size 1e-6, training and validation MSE for every epoch
# ----------------------------------------------------------------------------
w_a, w_hist_a, train_a, valid_a = train_sgd(1e-6)
print("\nPart (a), step size = 1e-6")
print(f"  final weights: w0 = {w_a[0]:.4f}, w1 = {w_a[1]:.4f}")
print(f"  epoch 1:    train MSE = {train_a[0]:.4f}, valid MSE = {valid_a[0]:.4f}")
print(f"  epoch {N_EPOCHS}: train MSE = {train_a[-1]:.4f}, valid MSE = {valid_a[-1]:.4f}")
 
epochs = np.arange(1, len(train_a) + 1)
plt.figure(figsize=(7, 4.5))
plt.plot(epochs, train_a, label="Training MSE")
plt.plot(epochs, valid_a, "--", label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Part (a): SGD with step size 1e-6")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "a_mse_step_1e-6.png"), dpi=200)
plt.show()

