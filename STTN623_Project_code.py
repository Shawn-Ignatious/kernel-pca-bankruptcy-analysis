import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo

# ----------------------------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------------------------
print("Fetching Taiwanese Bankruptcy Prediction dataset...")

dataset = fetch_ucirepo(id=572)

X = dataset.data.features
y = dataset.data.targets.values.ravel()


X.columns = X.columns.str.strip()

print("Full dataset shape:", X.shape)
print("Full class counts:", np.bincount(y))

# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------

X = X.drop(columns="Net Income Flag")
print("Shape after dropping 'Net Income Flag':", X.shape)


print("\nPreprocessing diagnostics (full data, before capping)")


X_range_sorted = (X.max() - X.min()).sort_values()
print("Smallest ranges:\n", X_range_sorted.head(3))
print("Largest ranges:\n", X_range_sorted.tail(3))


abs_skew = X.apply(skew).abs()
print(f"Median |skew|: {abs_skew.median():.2f}")
print("Variables with |skew| > 2:", (abs_skew > 2).sum())
print("Variables with |skew| > 5:", (abs_skew > 5).sum())
print("Most skewed variables:\n", abs_skew.sort_values(ascending=False).head())


X_std_only = StandardScaler().fit_transform(X.to_numpy(dtype=float))
abs_z = np.abs(X_std_only)
print(f"Largest absolute z-score (no capping): {abs_z.max():.1f}")
print("Variables with a z-score above 20:", (abs_z.max(axis=0) > 20).sum())



def preprocess(df, lower_q=0.01, upper_q=0.99, verbose=False):
    """
    1. Cap continuous variables at the 1st and 99th percentiles.
       Binary variables are left alone: capping a flag with only a few 1s
       would turn the whole column into zeros.
    2. Drop any variable that is constant after capping.
    3. Standardise (mean 0, sd 1), because capping changes means and spreads.

    Returns the scaled array and the list of retained column names.
    """
    is_binary = df.nunique() <= 2
    cont_cols = df.columns[~is_binary]

    df_cap = df.copy()
    lower = df[cont_cols].quantile(lower_q)
    upper = df[cont_cols].quantile(upper_q)
    df_cap[cont_cols] = df[cont_cols].clip(lower=lower, upper=upper, axis=1)

    const_cols = df_cap.columns[df_cap.std() == 0]
    if len(const_cols) > 0:
        if verbose:
            print("Constant columns removed after capping:", list(const_cols))
        df_cap = df_cap.drop(columns=const_cols)

    X_scaled = StandardScaler().fit_transform(df_cap.to_numpy(dtype=float))
    return X_scaled, list(df_cap.columns)



X_scaled_full, kept_full = preprocess(X, verbose=True)

skew_cap = pd.DataFrame(X_scaled_full).apply(skew).abs()
print("\n--- After capping and re-standardising (full data) ---")
print(f"Largest absolute z-score: {np.abs(X_scaled_full).max():.1f}")
print(f"Median |skew|: {skew_cap.median():.2f}")
print("Binary column 'Liability-Assets Flag' counts:")
print(X["Liability-Assets Flag"].value_counts())
print(f"Max |column mean|: {np.abs(X_scaled_full.mean(axis=0)).max():.2e}")
print("All column sds equal 1:", np.allclose(X_scaled_full.std(axis=0), 1))
print("Shape:", X_scaled_full.shape)

# ----------------------------------------------------------------------
# 3. Create balanced exploratory subsample
# ----------------------------------------------------------------------
rng = np.random.default_rng(42)

bankrupt_idx = np.where(y == 1)[0]
solvent_idx = np.where(y == 0)[0]

sampled_solvent_idx = rng.choice(
    solvent_idx,
    size=len(bankrupt_idx),
    replace=False
)

balanced_idx = np.concatenate([bankrupt_idx, sampled_solvent_idx])
balanced_idx = rng.permutation(balanced_idx)

X_bal = X.iloc[balanced_idx].copy()
y_bal = y[balanced_idx]

print("\nBalanced dataset shape:", X_bal.shape)
print("Balanced class counts:", np.bincount(y_bal))

# ----------------------------------------------------------------------
# 4. Preprocess the balanced subsample (cap, drop constants, standardise)
# ----------------------------------------------------------------------
# Percentile limits are computed on the data that PCA is actually run on.
X_scaled, kept_columns = preprocess(X_bal, verbose=True)
print("Features retained for PCA:", X_scaled.shape[1])

# ----------------------------------------------------------------------
# 5. Linear PCA
# ----------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_variance = pca.explained_variance_ratio_

print("--- Linear PCA Diagnostics ---")
print(f"PC1 variance: {pca_variance[0]*100:.2f}%")
print(f"PC2 variance: {pca_variance[1]*100:.2f}%")
print(
    "Total cumulative explained variance: "
    f"{np.sum(pca_variance)*100:.2f}%"
)

# ----------------------------------------------------------------------
# 6. Kernel PCA using RBF kernel
# ----------------------------------------------------------------------
gamma_value = 1 / X_scaled.shape[1]

kpca = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=gamma_value,
    random_state=42
)

X_kpca = kpca.fit_transform(X_scaled)

kernel_eigenvalues = kpca.eigenvalues_
kernel_power = kernel_eigenvalues / np.sum(kernel_eigenvalues)

print("--- Kernel PCA Diagnostics ---")
print(f"Gamma: {gamma_value:.4f}")
print(f"Kernel PC1 relative power: {kernel_power[0]*100:.2f}%")
print(f"Kernel PC2 relative power: {kernel_power[1]*100:.2f}%")

# ----------------------------------------------------------------------
# 7. Silhouette-score diagnostics
# ----------------------------------------------------------------------
silhouette_pca = silhouette_score(X_pca, y_bal)
silhouette_kpca = silhouette_score(X_kpca, y_bal)

print("--- Quantitative Results ---")
print(f"Linear PCA silhouette score: {silhouette_pca:.4f}")
print(f"Kernel PCA silhouette score: {silhouette_kpca:.4f}")

# ----------------------------------------------------------------------
# 8. Figure 1: Linear PCA versus Kernel PCA
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = {0: "#1f77b4", 1: "#d62728"}
labels = {0: "Solvent", 1: "Bankrupt"}

for class_value in [0, 1]:
    mask = y_bal == class_value

    axes[0].scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=colors[class_value],
        label=labels[class_value],
        alpha=0.65,
        edgecolors="k",
        linewidths=0.4
    )

    axes[1].scatter(
        X_kpca[mask, 0],
        X_kpca[mask, 1],
        c=colors[class_value],
        label=labels[class_value],
        alpha=0.65,
        edgecolors="k",
        linewidths=0.4
    )

axes[0].set_title("Linear PCA")
axes[0].set_xlabel(f"PC1 ({pca_variance[0]*100:.2f}% variance)")
axes[0].set_ylabel(f"PC2 ({pca_variance[1]*100:.2f}% variance)")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend()

axes[1].set_title(
    f"RBF Kernel PCA\nGamma = {gamma_value:.4f}"
)
axes[1].set_xlabel("Kernel PC 1")
axes[1].set_ylabel("Kernel PC 2")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend()

plt.suptitle(
    "Linear PCA versus Kernel PCA:\nTaiwanese Bankruptcy Prediction Dataset",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("figure1_pca_vs_kernel_pca.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 9. Figure 2: Gamma sensitivity
# ----------------------------------------------------------------------

gamma_values = sorted({0.001, 0.005, round(gamma_value, 4), 0.05, 0.1})

fig, axes = plt.subplots(1, len(gamma_values), figsize=(20, 4))

for ax, g in zip(axes, gamma_values):
    kpca_gamma = KernelPCA(
        n_components=2,
        kernel="rbf",
        gamma=g,
        random_state=42
    )

    X_kpca_gamma = kpca_gamma.fit_transform(X_scaled)

    for class_value in [0, 1]:
        mask = y_bal == class_value

        ax.scatter(
            X_kpca_gamma[mask, 0],
            X_kpca_gamma[mask, 1],
            c=colors[class_value],
            label=labels[class_value],
            alpha=0.6,
            s=18,
            edgecolors="none"
        )

    ax.set_title(f"Gamma = {g}")
    ax.set_xticks([])
    ax.set_yticks([])

axes[0].set_ylabel("Kernel PC 2")

fig.suptitle(
    "Sensitivity of RBF Kernel PCA to Gamma",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("figure2_gamma_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 10. Robustness to downsampling seed
# ----------------------------------------------------------------------
n_repeats = 10

silhouette_pca_scores = []
silhouette_kpca_scores = []

for seed in range(n_repeats):
    rng_seed = np.random.default_rng(seed)

    bankrupt_idx = np.where(y == 1)[0]
    solvent_idx = np.where(y == 0)[0]

    sampled_solvent_idx = rng_seed.choice(
        solvent_idx,
        size=len(bankrupt_idx),
        replace=False
    )

    balanced_idx = np.concatenate([bankrupt_idx, sampled_solvent_idx])
    balanced_idx = rng_seed.permutation(balanced_idx)

    X_bal_seed = X.iloc[balanced_idx].copy()
    y_bal_seed = y[balanced_idx]

    # Same preprocessing as the main analysis, fitted on this subsample
    X_scaled_seed, _ = preprocess(X_bal_seed)

    # Linear PCA
    pca_seed = PCA(n_components=2, random_state=42)
    X_pca_seed = pca_seed.fit_transform(X_scaled_seed)

    # RBF Kernel PCA
    gamma_seed = 1 / X_scaled_seed.shape[1]
    kpca_seed = KernelPCA(
        n_components=2,
        kernel="rbf",
        gamma=gamma_seed,
        random_state=42
    )
    X_kpca_seed = kpca_seed.fit_transform(X_scaled_seed)

    silhouette_pca_scores.append(
        silhouette_score(X_pca_seed, y_bal_seed)
    )

    silhouette_kpca_scores.append(
        silhouette_score(X_kpca_seed, y_bal_seed)
    )

# Summary statistics
pca_mean = np.mean(silhouette_pca_scores)
pca_std = np.std(silhouette_pca_scores, ddof=1)

kpca_mean = np.mean(silhouette_kpca_scores)
kpca_std = np.std(silhouette_kpca_scores, ddof=1)

print("--- Robustness to Downsampling Seed ---")
print(f"Linear PCA silhouette scores: {silhouette_pca_scores}")
print(f"Kernel PCA silhouette scores: {silhouette_kpca_scores}")

print(f"Linear PCA mean: {pca_mean:.4f}")
print(f"Linear PCA standard deviation: {pca_std:.4f}")

print(f"Kernel PCA mean: {kpca_mean:.4f}")
print(f"Kernel PCA standard deviation: {kpca_std:.4f}")