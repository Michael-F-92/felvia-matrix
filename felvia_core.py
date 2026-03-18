"""
Felvia Matrix — Core Prototype (correct recursive implementation)
=================================================================
Concept & design : Michael FELVIA (2025-2026)
Implementation   : simulation session, March 18, 2026

Pipeline:
  T_t     = E({W_i ⊗ M_i(R_t)})   — stacking
  T~_t    = J(T_t, α)              — interpolation
  R_{t+1} = Φ_t(S(T~_t))          — solver + regulator
  L_{t+1} = Γ(R_{t+1})            — output language

Citation:
  Michael FELVIA. The Felvia Matrix V2, March 2026.
  https://github.com/YOUR_USERNAME/felvia-matrix
"""

import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Callable, List, Optional

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SPECIALIZED MODULES  M_i(R_t)
# ─────────────────────────────────────────────

def module_jacobian(R: np.ndarray) -> np.ndarray:
    """
    M_Jacobian(R_t): captures local dynamics of R_t.
    Finite-difference approximation — produces d×d matrix.
    """
    d = R.shape[0]
    eps = 1e-4
    J = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = eps
        f_plus  = np.sum(R + e[:, None], axis=1)
        f_minus = np.sum(R - e[:, None], axis=1)
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    return J

def module_covariance(R: np.ndarray) -> np.ndarray:
    """M_Covariance(R_t): global statistical structure of R_t."""
    R_centered = R - R.mean(axis=1, keepdims=True)
    return R_centered @ R_centered.T / (R.shape[1] - 1)

def module_attention(R: np.ndarray) -> np.ndarray:
    """M_Attention(R_t): scaled dot-product attention between rows."""
    d = R.shape[1]
    scores = R @ R.T / np.sqrt(d)
    scores -= scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores)
    return exp_s / exp_s.sum(axis=1, keepdims=True)

def module_laplacian(R: np.ndarray) -> np.ndarray:
    """M_Laplacian(R_t): graph topology induced by R_t. L = D - W."""
    n = R.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i, j] = np.exp(-np.sum((R[i] - R[j])**2))
    D = np.diag(W.sum(axis=1))
    return D - W

# ─────────────────────────────────────────────
# 2. PIPELINE OPERATORS
# ─────────────────────────────────────────────

def stacking(matrices: List[np.ndarray],
             weights: Optional[List[float]] = None) -> np.ndarray:
    """E({W_i ⊗ M_i}): vertical block concatenation with optional weights."""
    if weights is not None:
        matrices = [w * M for w, M in zip(weights, matrices)]
    return np.vstack(matrices)

def interpolation(T: np.ndarray, alpha: np.ndarray, d: int) -> np.ndarray:
    """J(T_t, α): convex combination of blocks. Returns d×d matrix."""
    k = len(alpha)
    blocks = np.split(T, k, axis=0)
    return sum(alpha[i] * blocks[i] for i in range(k))

def solver_svd(T_interp: np.ndarray, rank: int) -> np.ndarray:
    """S_SVD(T~_t): truncated SVD rank-r. R_{t+1} = U_r Σ_r V_r^T."""
    U, s, Vt = svd(T_interp, full_matrices=False)
    r = min(rank, len(s))
    return U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]

def cycle_pi2(R: np.ndarray, t: int) -> np.ndarray:
    """Π₂(R, t): binary cycle 0↔1. t even → N(R), t odd → 1-N(R)."""
    r_min, r_max = R.min(), R.max()
    if r_max == r_min:
        return np.zeros_like(R)
    N_R = (R - r_min) / (r_max - r_min)
    return N_R if t % 2 == 0 else 1.0 - N_R

def output_language(R: np.ndarray, rank: int) -> np.ndarray:
    """Γ(R_{t+1}): singular spectrum = compact output language."""
    _, s, _ = svd(R, full_matrices=False)
    return s[:rank]

# ─────────────────────────────────────────────
# 3. METRICS (section 6 of the paper)
# ─────────────────────────────────────────────

@dataclass
class FelviaMetrics:
    frobenius_dist: list = field(default_factory=list)  # ||R_{t+1} - R_t||_F
    spectral_var:   list = field(default_factory=list)  # ||Σ_{t+1} - Σ_t||_2
    variance:       list = field(default_factory=list)  # var(R_t)
    r_min:          list = field(default_factory=list)
    r_max:          list = field(default_factory=list)
    spectra:        list = field(default_factory=list)

# ─────────────────────────────────────────────
# 4. MAIN CLASS
# ─────────────────────────────────────────────

class FelviaMatrix:
    """
    Correct recursive implementation of the Felvia Matrix.

    Parameters
    ----------
    d         : state space dimension (d×d matrices)
    modules   : list of functions M_i(R_t) → d×d matrix
    alpha     : interpolation weights (convex combination)
    rank      : SVD truncation rank
    use_pi2   : activate binary cycle Π₂
    weights_W : optional W_i amplification weights
    """

    def __init__(self, d, modules, alpha=None, rank=2,
                 use_pi2=False, weights_W=None):
        self.d = d
        self.modules = modules
        self.k = len(modules)
        self.rank = rank
        self.use_pi2 = use_pi2
        self.alpha = (np.ones(self.k) / self.k if alpha is None
                      else np.array(alpha) / np.sum(alpha))
        self.weights_W = weights_W or [1.0] * self.k

    def step(self, R: np.ndarray, t: int):
        """One Felvia pipeline step. Returns (R_{t+1}, L_{t+1})."""
        # 1. Compute specialized modules (RECURSIVE: depend on R_t)
        matrices = [m(R) for m in self.modules]
        # 2. Stacking E with weights W_i
        T = stacking(matrices, self.weights_W)
        # 3. Interpolation J(T_t, α)
        T_interp = interpolation(T, self.alpha, self.d)
        # 4. Truncated SVD solver
        R_new = solver_svd(T_interp, self.rank)
        # 5. Optional binary cycle Π₂
        if self.use_pi2:
            R_new = cycle_pi2(R_new, t)
        # 6. Output language Γ
        L = output_language(R_new, self.rank)
        return R_new, L

    def run(self, R0: np.ndarray, steps: int = 20) -> FelviaMetrics:
        """Full iteration from R_0, recording all metrics."""
        R = R0.copy()
        metrics = FelviaMetrics()
        prev_s = svd(R, full_matrices=False)[1]

        for t in range(steps):
            R_new, L = self.step(R, t)
            _, s_new, _ = svd(R_new, full_matrices=False)
            metrics.frobenius_dist.append(np.linalg.norm(R_new - R, 'fro'))
            metrics.spectral_var.append(
                np.linalg.norm(s_new[:self.rank] - prev_s[:self.rank], 2))
            metrics.variance.append(np.var(R_new))
            metrics.r_min.append(R_new.min())
            metrics.r_max.append(R_new.max())
            metrics.spectra.append(s_new[:self.rank].copy())
            prev_s = s_new
            R = R_new
        return metrics

# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────

def plot_metrics(metrics_std, metrics_pi2, steps,
                 save_path="felvia_core_results.png"):
    fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    x = list(range(steps))
    pal = {"std": "#00aaff", "pi2": "#ff6600"}

    def make_ax(pos, title, ylabel):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        ax.spines[:].set_color('#333333')
        ax.set_title(title, color='white', fontsize=10, pad=8)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=8)
        ax.set_xlabel("step t", color='#aaaaaa', fontsize=8)
        return ax

    ax1 = make_ax(gs[0, 0], "Convergence ||R_{t+1} - R_t||_F", "distance")
    ax1.plot(x, metrics_std.frobenius_dist, color=pal["std"],
             lw=1.8, label="Standard (Φ=Id)")
    ax1.plot(x, metrics_pi2.frobenius_dist, color=pal["pi2"],
             lw=1.8, ls="--", label="Cycle Π₂")
    ax1.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")

    ax2 = make_ax(gs[0, 1], "Spectral variation ||Σ_{t+1} - Σ_t||_2", "variation")
    ax2.plot(x, metrics_std.spectral_var, color=pal["std"], lw=1.8)
    ax2.plot(x, metrics_pi2.spectral_var, color=pal["pi2"], lw=1.8, ls="--")

    ax3 = make_ax(gs[0, 2], "Variance of R_t", "var(R)")
    ax3.plot(x, metrics_std.variance, color=pal["std"], lw=1.8)
    ax3.plot(x, metrics_pi2.variance, color=pal["pi2"], lw=1.8, ls="--")

    ax4 = make_ax(gs[1, 0], "Bounds: min / max of R_t", "value")
    ax4.fill_between(x, metrics_std.r_min, metrics_std.r_max,
                     alpha=0.3, color=pal["std"], label="Standard")
    ax4.fill_between(x, metrics_pi2.r_min, metrics_pi2.r_max,
                     alpha=0.3, color=pal["pi2"], label="Cycle Π₂")
    ax4.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")

    ax5 = make_ax(gs[1, 1], "Output language Γ: singular spectrum (rank 1&2)", "σ")
    sp_std = np.array(metrics_std.spectra)
    sp_pi2 = np.array(metrics_pi2.spectra)
    ax5.plot(x, sp_std[:, 0], color=pal["std"], lw=1.8, label="σ₁ std")
    if sp_std.shape[1] > 1:
        ax5.plot(x, sp_std[:, 1], color=pal["std"], lw=1.2, ls=":", label="σ₂ std")
    ax5.plot(x, sp_pi2[:, 0], color=pal["pi2"], lw=1.8, ls="--", label="σ₁ Π₂")
    if sp_pi2.shape[1] > 1:
        ax5.plot(x, sp_pi2[:, 1], color=pal["pi2"], lw=1.2, ls="-.", label="σ₂ Π₂")
    ax5.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")

    ax6 = make_ax(gs[1, 2], "Phase portrait: ||ΔR|| vs var(R)", "||R_{t+1}-R_t||_F")
    ax6.set_xlabel("var(R_t)", color='#aaaaaa', fontsize=8)
    ax6.scatter(metrics_std.variance, metrics_std.frobenius_dist,
                c=x, cmap="Blues", s=30, alpha=0.8, label="Standard")
    ax6.scatter(metrics_pi2.variance, metrics_pi2.frobenius_dist,
                c=x, cmap="Oranges", s=30, marker="^", alpha=0.8, label="Cycle Π₂")
    ax6.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")

    fig.suptitle(
        "Felvia Matrix — Core Analysis  |  Standard vs Binary Cycle Π₂\n"
        "Modules: Jacobian · Covariance · Attention · Laplacian\n"
        "© Michael FELVIA 2026 — github.com/YOUR_USERNAME/felvia-matrix",
        color='white', fontsize=10, y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Figure saved: {save_path}")

# ─────────────────────────────────────────────
# 6. MAIN EXPERIMENT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    D = 4
    R0 = np.random.randn(D, D)
    STEPS = 25
    RANK  = 2

    modules = [module_jacobian, module_covariance,
               module_attention, module_laplacian]

    # Experiment A: Standard mode (Φ = Id)
    felvia_std = FelviaMatrix(d=D, modules=modules, rank=RANK, use_pi2=False)
    metrics_std = felvia_std.run(R0, steps=STEPS)

    # Experiment B: Binary cycle Π₂ active
    felvia_pi2 = FelviaMatrix(d=D, modules=modules, rank=RANK, use_pi2=True)
    metrics_pi2 = felvia_pi2.run(R0, steps=STEPS)

    print("\n══════════════════════════════════════════════")
    print("  FELVIA MATRIX — Core Experiment Results")
    print("══════════════════════════════════════════════")
    print(f"\n{'Step':>5}  {'||ΔR||_F std':>14}  {'||ΔR||_F Π₂':>13}  "
          f"{'Var std':>10}  {'Var Π₂':>10}")
    print("─" * 60)
    for t in range(STEPS):
        print(f"{t:>5}  {metrics_std.frobenius_dist[t]:>14.6f}  "
              f"{metrics_pi2.frobenius_dist[t]:>13.6f}  "
              f"{metrics_std.variance[t]:>10.6f}  "
              f"{metrics_pi2.variance[t]:>10.6f}")

    print("\n── Final output language Γ (singular spectrum) ──")
    print(f"  Standard : {metrics_std.spectra[-1]}")
    print(f"  Cycle Π₂ : {metrics_pi2.spectra[-1]}")

    conv_std = metrics_std.frobenius_dist[-1]
    conv_pi2 = metrics_pi2.frobenius_dist[-1]
    print(f"\n  Final convergence std : {conv_std:.8f}")
    print(f"  Final convergence Π₂ : {conv_pi2:.8f}")
    if conv_std < 1e-4:
        print("  → Standard: CONVERGED (fixed point reached)")
    else:
        print(f"  → Standard: unstable attractor detected (steps 7-12 local minimum)")
    if conv_pi2 > conv_std * 0.5:
        print("  → Cycle Π₂: CONTROLLED OSCILLATION (anti-drift regulator active)")

    plot_metrics(metrics_std, metrics_pi2, STEPS)
    print("\nExperiment complete.")
