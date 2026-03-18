"""
Felvia Matrix — Hypothesis Test: LLM + World Model Fusion
==========================================================
Concept & design : Michael FELVIA (2025-2026)
Implementation   : simulation session, March 18, 2026

Test: can Felvia produce a representation that captures information
from both LLM modules and WM modules simultaneously?

Metrics:
  - Spectral entropy (richness)
  - Predictive power ||R - R_world||_F
  - Information coverage (correlation with source modules)
  - Dynamic stability ||ΔR||_F

Citation:
  Michael FELVIA. The Felvia Matrix V2, March 2026.
  https://github.com/YOUR_USERNAME/felvia-matrix
"""

import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)
D = 8

# ─────────────────────────────────────────────
# LLM MODULES (no world access — reason on R_t)
# ─────────────────────────────────────────────

def module_semantic(R):
    d = R.shape[0]
    s = R @ R.T / np.sqrt(d); s -= s.max(axis=1, keepdims=True)
    e = np.exp(s); return e / e.sum(axis=1, keepdims=True)

def module_causal(R):
    d = R.shape[0]; eps = 1e-4; J = np.zeros((d, d))
    for i in range(d):
        ei = np.zeros(d); ei[i] = eps
        J[:, i] = (np.tanh(R + ei[:, None]).sum(1) -
                   np.tanh(R - ei[:, None]).sum(1)) / (2 * eps)
    return J

def module_memory(R):
    Rc = R - R.mean(axis=1, keepdims=True)
    return Rc @ Rc.T / max(R.shape[1] - 1, 1)

# ─────────────────────────────────────────────
# WORLD MODEL MODULES (noisy sensory access)
# ─────────────────────────────────────────────

def module_state(R, t, noise=0.15):
    d = R.shape[0]
    true_state = (np.sin(2*np.pi*t/10) * np.eye(d) +
                  np.cos(2*np.pi*t/7)  * np.ones((d, d)) / d)
    return true_state + noise * np.random.randn(d, d)

def module_dynamics(R, t, noise=0.15):
    return module_state(R, t, noise) - R

def module_uncertainty(R, t, noise=0.15):
    error = module_state(R, t, noise) - R
    return error @ error.T / D + 0.01 * np.eye(D)

# ─────────────────────────────────────────────
# FELVIA PIPELINE
# ─────────────────────────────────────────────

def felvia_fusion(R, t, alpha_llm=0.5, rank=3, use_pi2=False):
    d = R.shape[0]
    alpha_wm = 1.0 - alpha_llm
    llm = {"sem": module_semantic(R),
           "cau": module_causal(R),
           "mem": module_memory(R)}
    wm  = {"sta": module_state(R, t),
           "dyn": module_dynamics(R, t),
           "unc": module_uncertainty(R, t)}
    n_llm, n_wm = 3, 3
    wl, ww = alpha_llm / n_llm, alpha_wm / n_wm
    weighted = ([wl * M for M in llm.values()] +
                [ww * M for M in wm.values()])
    T = np.vstack(weighted)
    blocks = np.split(T, 6, axis=0)
    T_interp = sum(blocks)
    U, s, Vt = svd(T_interp, full_matrices=False)
    r = min(rank, len(s))
    R_new = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
    if use_pi2:
        mn, mx = R_new.min(), R_new.max()
        if mx > mn:
            N = (R_new - mn) / (mx - mn)
            R_new = N if t % 2 == 0 else 1.0 - N
    _, s_out, _ = svd(R_new, full_matrices=False)
    return R_new, s_out[:rank], llm, wm

# ─────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────

def spectral_entropy(M):
    _, s, _ = svd(M, full_matrices=False)
    s = s[s > 1e-10]; p = s / s.sum()
    return float(scipy_entropy(p))

def information_coverage(L_fused, source_matrices):
    _, s_f, _ = svd(L_fused, full_matrices=False)
    scores = []
    for M in source_matrices:
        _, s_m, _ = svd(M, full_matrices=False)
        n = min(len(s_f), len(s_m))
        if n == 0 or s_m[:n].std() == 0 or s_f[:n].std() == 0:
            scores.append(0.0); continue
        c = np.corrcoef(s_f[:n], s_m[:n])[0, 1]
        scores.append(abs(c) if not np.isnan(c) else 0.0)
    return float(np.mean(scores))

# ─────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────

def run_experiment(steps=40):
    R0 = np.random.randn(D, D) * 0.5
    results = {k: {"entropy": [], "pred_err": [], "coverage": [], "frob": []}
               for k in ["llm_only", "wm_only", "felvia_50", "felvia_pi2"]}
    R = {k: R0.copy() for k in results}
    R_prev = R0.copy()

    for t in range(steps):
        cfgs = [("llm_only", 1.0, False), ("wm_only", 0.0, False),
                ("felvia_50", 0.5, False), ("felvia_pi2", 0.5, True)]
        for name, al, pi2 in cfgs:
            R_new, _, llm_m, wm_m = felvia_fusion(R[name], t, alpha_llm=al,
                                                    use_pi2=pi2)
            R_true_next = (np.sin(2*np.pi*(t+1)/10)*np.eye(D) +
                           np.cos(2*np.pi*(t+1)/7)*np.ones((D,D))/D)
            results[name]["entropy"].append(spectral_entropy(R_new))
            results[name]["pred_err"].append(
                np.linalg.norm(R_new - R_true_next, 'fro'))
            results[name]["coverage"].append(
                information_coverage(R_new,
                    list(llm_m.values()) + list(wm_m.values())))
            results[name]["frob"].append(
                np.linalg.norm(R_new - R_prev, 'fro'))
            R[name] = R_new
        R_prev = R_new
    return results, steps

def plot_results(results, steps, path="felvia_llm_wm_results.png"):
    palette = {
        "llm_only":   ("#4488ff", "-",  "LLM only (α=1)"),
        "wm_only":    ("#ff4444", "-",  "WM only (α=0)"),
        "felvia_50":  ("#00ff99", "-",  "Felvia 50/50"),
        "felvia_pi2": ("#ffcc00", "--", "Felvia 50/50 + Π₂"),
    }
    fig = plt.figure(figsize=(18, 11), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.38)
    x   = list(range(steps))
    metrics = [
        ("entropy",  "Spectral richness (entropy of Γ)",          "Shannon entropy",        gs[0, 0:2]),
        ("pred_err", "Predictive error ||R - R_world||_F",        "error (↓ better)",       gs[0, 2:4]),
        ("coverage", "Information coverage (LLM + WM sources)",   "spectral correlation",   gs[1, 0:2]),
        ("frob",     "Dynamic stability ||ΔR||_F",                 "Frobenius distance",     gs[1, 2:4]),
    ]
    for metric, title, ylabel, pos in metrics:
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors='#aaaaaa', labelsize=8)
        ax.spines[:].set_color('#444444')
        ax.set_title(title, color='white', fontsize=10, pad=8)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=8)
        ax.set_xlabel("step t", color='#aaaaaa', fontsize=8)
        for name, (color, ls, label) in palette.items():
            ax.plot(x, results[name][metric], color=color, lw=2.0,
                    linestyle=ls, label=label, alpha=0.9)
        ax.legend(fontsize=7.5, facecolor="#111111",
                  labelcolor="white", framealpha=0.8)
    fig.suptitle(
        "Felvia Matrix — Hypothesis Test: LLM + World Model Fusion\n"
        "Question: does Felvia fusion produce a representation superior to each source?\n"
        "© Michael FELVIA 2026 — github.com/YOUR_USERNAME/felvia-matrix",
        color='white', fontsize=11, y=0.99)
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(); print(f"Figure saved: {path}")

def print_verdict(results, steps):
    half = steps // 2
    summary = {k: {m: np.mean(results[k][m][half:]) for m in results[k]}
               for k in results}
    print("\n══════════════════════════════════════════════")
    print("  VERDICT — LLM + World Model Fusion Test")
    print("══════════════════════════════════════════════")
    print(f"\n{'':20} {'Entropy':>10} {'Pred err':>12} "
          f"{'Coverage':>12} {'Stability':>10}")
    print("─" * 68)
    labels = {"llm_only": "LLM only", "wm_only": "WM only",
              "felvia_50": "Felvia 50/50", "felvia_pi2": "Felvia + Π₂"}
    for k, lb in labels.items():
        s = summary[k]
        print(f"  {lb:18} {s['entropy']:>10.4f} {s['pred_err']:>12.4f} "
              f"{s['coverage']:>12.4f} {s['frob']:>10.4f}")
    felvia_cov = summary["felvia_50"]["coverage"]
    llm_cov    = summary["llm_only"]["coverage"]
    wm_cov     = summary["wm_only"]["coverage"]
    if felvia_cov > max(llm_cov, wm_cov):
        print("\n✅ Coverage: Felvia captures LLM AND WM information simultaneously")
    else:
        print("\n⚠️  Coverage: partial — fixed alpha dilutes specialist information")
    print("\nConclusion: fixed alpha is context-blind.")
    print("→ Next step: adaptive alpha (see felvia_adaptive_v2.py)")
    print("══════════════════════════════════════════════")

if __name__ == "__main__":
    print("Felvia Matrix — LLM + WM Fusion Test | 40 steps | D=8")
    results, steps = run_experiment(steps=40)
    print_verdict(results, steps)
    plot_results(results, steps)
    print("\nExperiment complete.")
