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

# ----------------------------------------------------------------------------
# Part (b): try different step sizes, pick the best by validation MSE
# ----------------------------------------------------------------------------
results = {}
print("\nPart (b): step size sweep")
print(f"  {'step size':>10} | {'w0':>8} {'w1':>8} | {'train MSE':>10} {'valid MSE':>10}")
for lr in STEP_SIZES:
    w, w_hist, tr, va = train_sgd(lr)
    results[lr] = (w, w_hist, tr, va)
    final_va = va[-1]
    if np.isfinite(final_va):
        print(f"  {lr:>10.0e} | {w[0]:>8.4f} {w[1]:>8.4f} | {tr[-1]:>10.5f} {final_va:>10.5f}")
    else:
        print(f"  {lr:>10.0e} | diverged after {len(va)} epoch(s), weights blew up to infinity")
 
# best = lowest final validation MSE (diverged runs are ignored)
valid_final = {lr: r[3][-1] for lr, r in results.items() if np.isfinite(r[3][-1])}
best_lr = min(valid_final, key=valid_final.get)
w_best = results[best_lr][0]
test_mse = mse(w_best, x_test, y_test)
print(f"\n  Best step size: {best_lr:.0e}")
print(f"  Chosen model: y = {w_best[0]:.4f} + {w_best[1]:.4f} x")
print(f"  Validation MSE = {valid_final[best_lr]:.5f}")
print(f"  TEST MSE       = {test_mse:.5f}")
 
# Sanity check: exact least-squares solution (no gradient descent)
w_exact = np.linalg.lstsq(design_matrix(x_train), y_train, rcond=None)[0]
print(f"\n  Check, exact least squares: y = {w_exact[0]:.4f} + {w_exact[1]:.4f} x")
print(f"  valid MSE = {mse(w_exact, x_valid, y_valid):.5f}, "
      f"test MSE = {mse(w_exact, x_test, y_test):.5f}")
 
# Validation MSE vs epoch for every step size that did not diverge
plt.figure(figsize=(7, 4.5))
for lr, (w, w_hist, tr, va) in results.items():
    if np.all(np.isfinite(va)):
        plt.plot(np.arange(1, len(va) + 1), va, label=f"step = {lr:.0e}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Epoch (log scale)")
plt.ylabel("Validation MSE (log scale)")
plt.title("Part (b): validation MSE for different step sizes")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "b_valid_mse_step_sizes.png"), dpi=200)
plt.show()
 
# Training + validation MSE for the chosen step size
_, _, tr_b, va_b = results[best_lr]
plt.figure(figsize=(7, 4.5))
plt.plot(np.arange(1, len(tr_b) + 1), tr_b, label="Training MSE")
plt.plot(np.arange(1, len(va_b) + 1), va_b, "--", label="Validation MSE")
plt.xscale("log")
plt.xlabel("Epoch (log scale)")
plt.ylabel("MSE")
plt.title(f"Part (b): chosen step size {best_lr:.0e}")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "b_mse_best_step.png"), dpi=200)
plt.show()
 

# ----------------------------------------------------------------------------
# Part (c): how the fit evolves during training (chosen step size)
# ----------------------------------------------------------------------------
w_hist_best = results[best_lr][1]
x_line = np.linspace(min(x_train.min(), x_valid.min()),
                     max(x_train.max(), x_valid.max()), 100)
 
# Optional: one plot for every single epoch
if SAVE_EVERY_EPOCH_FRAMES:
    frame_dir = os.path.join(FIG_DIR, "every_epoch")
    os.makedirs(frame_dir, exist_ok=True)
    for e, w in enumerate(w_hist_best, start=1):
        plt.figure(figsize=(5, 4))
        plt.scatter(x_train, y_train, s=8, alpha=0.5, label="Training data")
        plt.plot(x_line, w[0] + w[1] * x_line, "r", lw=2, label="Fit")
        plt.title(f"Epoch {e}")
        plt.xlabel("x"); plt.ylabel("y"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(frame_dir, f"epoch_{e:04d}.png"), dpi=100)
        plt.close()
 
# The 5 snapshots for the report
SNAPSHOT_EPOCHS = [1, 3, 10, 30, N_EPOCHS]
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
axes = axes.ravel()
axes[5].axis("off")    # only 5 snapshots, hide the 6th panel
for ax, e in zip(axes, SNAPSHOT_EPOCHS):
    w = w_hist_best[e - 1]
    ax.scatter(x_train, y_train, s=8, alpha=0.4, label="Training data")
    ax.plot(x_line, w[0] + w[1] * x_line, "r", lw=2, label="Fit")
    ax.set_title(f"Epoch {e}\ny = {w[0]:.2f} + {w[1]:.2f}x\n"
                 f"valid MSE = {mse(w, x_valid, y_valid):.3f}", fontsize=9)
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("y")
axes[3].set_ylabel("y")
axes[0].legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "c_fit_evolution.png"), dpi=200)
plt.show()
 
# Same idea with the slow step size from part (a), where the change is gradual
SNAPSHOT_EPOCHS_A = [1, 10, 100, 300, N_EPOCHS]
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
axes = axes.ravel()
axes[5].axis("off")    # only 5 snapshots, hide the 6th panel
for ax, e in zip(axes, SNAPSHOT_EPOCHS_A):
    w = w_hist_a[e - 1]
    ax.scatter(x_train, y_train, s=8, alpha=0.4, label="Training data")
    ax.plot(x_line, w[0] + w[1] * x_line, "r", lw=2, label="Fit")
    ax.set_title(f"Epoch {e}\ny = {w[0]:.2f} + {w[1]:.2f}x\n"
                 f"valid MSE = {mse(w, x_valid, y_valid):.3f}", fontsize=9)
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("y")
axes[3].set_ylabel("y")
axes[0].legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "c_fit_evolution_step_1e-6.png"), dpi=200)
plt.show()