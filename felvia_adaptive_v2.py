"""
Felvia Matrix — Adaptive Alpha Controller V2
============================================
Concept & design : Michael FELVIA (2025-2026)
Implementation   : simulation session, March 18, 2026

Core correction over fixed-alpha version:
  - All signals normalized by EMA (Exponential Moving Average)
  - Balanced alpha range [0.08, 0.92] — no systematic WM dominance
  - Π₂ triggered by normalized uncertainty (not raw trace)

Three contextual signals:
  δ_world       : world rate of change  → α_WM up when world is unstable
  σ_uncertainty : sensor noise level    → α_LLM up when WM is unreliable
  δ_LLM         : reasoning stability   → α_LLM up when LLM is stable

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
# MODULES
# ─────────────────────────────────────────────

def module_semantic(R):
    d = R.shape[0]; s = R @ R.T / np.sqrt(d)
    s -= s.max(axis=1, keepdims=True)
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

def world_state(t, D):
    return (np.sin(2*np.pi*t/10) * np.eye(D) +
            np.cos(2*np.pi*t/7)  * np.ones((D, D)) / D)

def module_state(R, t, noise=0.15):
    return world_state(t, D) + noise * np.random.randn(D, D)

def module_dynamics(R, t, noise=0.15):
    return module_state(R, t, noise) - R

def module_uncertainty(R, t, noise=0.15):
    err = module_state(R, t, noise) - R
    return err @ err.T / D + 0.01 * np.eye(D)

# ─────────────────────────────────────────────
# ADAPTIVE ALPHA CONTROLLER V2
# ─────────────────────────────────────────────

class AdaptiveAlphaV2:
    """
    EMA-normalized adaptive alpha controller.

    All raw signals are normalized by their exponential moving average,
    so a temporarily large world-change rate does not permanently
    dominate the alpha computation.

    Parameters
    ----------
    beta      : EMA smoothing factor (0.8 = slow adaptation)
    seuil_pi2 : Π₂ activation threshold on normalized uncertainty
    momentum  : alpha smoothing (prevents abrupt switches)
    """

    def __init__(self, beta=0.8, seuil_pi2=0.22,
                 alpha_min=0.05, alpha_max=0.95, momentum=0.3):
        self.beta = beta; self.seuil_pi2 = seuil_pi2
        self.alpha_min = alpha_min; self.alpha_max = alpha_max
        self.momentum = momentum
        self.ema_world = 1e-6; self.ema_llm = 1e-6; self.ema_uncert = 1e-6
        self.alpha_llm_prev = 0.5; self.sp = None; self.Ap = None

    def compute(self, R, t):
        """
        Returns (alpha_llm, alpha_wm, use_pi2, diagnostics).
        """
        d = R.shape[0]

        # Signal 1: world rate of change
        sn = world_state(t, d)
        dw = np.linalg.norm(sn - self.sp, 'fro') if self.sp is not None else 0.0
        self.sp = sn.copy()
        self.ema_world = self.beta * self.ema_world + (1 - self.beta) * dw
        norm_world = min(dw / (self.ema_world + 1e-9), 2.0) / 2.0

        # Signal 2: sensor uncertainty
        err = sn - R; cov = err @ err.T / d + 0.01 * np.eye(d)
        sigma = np.trace(cov) / d
        self.ema_uncert = self.beta * self.ema_uncert + (1 - self.beta) * sigma
        norm_uncert = min(sigma / (self.ema_uncert + 1e-9), 2.0) / 2.0
        wm_reliability = 1.0 - norm_uncert

        # Signal 3: LLM stability (attention variation)
        sc = R @ R.T / np.sqrt(d); sc -= sc.max(axis=1, keepdims=True)
        ex = np.exp(sc); A = ex / ex.sum(axis=1, keepdims=True)
        dl = np.linalg.norm(A - self.Ap, 'fro') if self.Ap is not None else 0.0
        self.Ap = A.copy()
        self.ema_llm = self.beta * self.ema_llm + (1 - self.beta) * dl
        norm_llm = min(dl / (self.ema_llm + 1e-9), 2.0) / 2.0
        llm_stability = 1.0 - norm_llm

        # Alpha computation
        score_wm  = 1.5 * (0.5 * norm_world + 0.5 * wm_reliability)
        score_llm = 1.5 * llm_stability
        aw_raw = np.clip(score_wm / (score_wm + score_llm + 1e-9),
                         self.alpha_min, self.alpha_max)

        # Momentum smoothing
        al = np.clip((1 - self.momentum) * (1 - aw_raw) +
                     self.momentum * self.alpha_llm_prev,
                     self.alpha_min, self.alpha_max)
        self.alpha_llm_prev = al
        aw = 1.0 - al
        use_pi2 = bool(norm_uncert > self.seuil_pi2)

        return float(al), float(aw), use_pi2, {
            "alpha_llm": al, "alpha_wm": aw,
            "norm_world": norm_world, "norm_uncert": norm_uncert,
            "norm_llm": norm_llm, "use_pi2": use_pi2
        }

# ─────────────────────────────────────────────
# FELVIA STEP WITH ADAPTIVE ALPHA
# ─────────────────────────────────────────────

def felvia_step(R, t, alpha_llm, alpha_wm, use_pi2=False, rank=3):
    n = 3; wl = alpha_llm / n; ww = alpha_wm / n
    mats = [module_semantic(R), module_causal(R), module_memory(R),
            module_state(R, t), module_dynamics(R, t), module_uncertainty(R, t)]
    ws = [wl, wl, wl, ww, ww, ww]
    T = np.vstack([w * M for w, M in zip(ws, mats)])
    T_interp = sum(np.split(T, 6, axis=0))
    U, s, Vt = svd(T_interp, full_matrices=False)
    r = min(rank, len(s))
    R_new = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
    if use_pi2:
        mn, mx = R_new.min(), R_new.max()
        if mx > mn:
            N = (R_new - mn) / (mx - mn)
            R_new = N if t % 2 == 0 else 1.0 - N
    _, so, _ = svd(R_new, full_matrices=False)
    return R_new, so[:rank]

def spectral_H(M):
    _, s, _ = svd(M, full_matrices=False); s = s[s > 1e-10]; p = s / s.sum()
    return float(scipy_entropy(p))

# ─────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────

def run(steps=60):
    R0 = np.random.randn(D, D) * 0.5
    Rs = {k: R0.copy() for k in ["llm", "wm", "fixed", "adapt"]}
    ctrl = AdaptiveAlphaV2()
    logs = {k: {"err": [], "ent": [], "frob": [], "al": [], "aw": [], "pi2": []}
            for k in Rs}
    R_prev = R0.copy()

    for t in range(steps):
        R_true = (np.sin(2*np.pi*(t+1)/10) * np.eye(D) +
                  np.cos(2*np.pi*(t+1)/7)  * np.ones((D, D)) / D)
        al, aw, pi2, _ = ctrl.compute(Rs["adapt"], t)
        for k, (a, b, p) in [("llm", (1., 0., False)), ("wm", (0., 1., False)),
                               ("fixed", (.5, .5, False)), ("adapt", (al, aw, pi2))]:
            Rn, _ = felvia_step(Rs[k], t, a, b, p)
            logs[k]["err"].append(np.linalg.norm(Rn - R_true, 'fro'))
            logs[k]["ent"].append(spectral_H(Rn))
            logs[k]["frob"].append(np.linalg.norm(Rn - R_prev, 'fro'))
            logs[k]["al"].append(a); logs[k]["aw"].append(b)
            logs[k]["pi2"].append(float(p))
            Rs[k] = Rn
        R_prev = R_true
    return logs, steps

def plot(logs, steps, path="felvia_adaptive_v2_results.png"):
    pal = {"llm":   ("#4488ff", "-",  "LLM only"),
           "wm":    ("#ff4444", "-",  "WM only"),
           "fixed": ("#aaaaaa", "--", "Felvia fixed 50/50"),
           "adapt": ("#00ff99", "-",  "Felvia ADAPTIVE α ★")}
    x = list(range(steps))
    fig = plt.figure(figsize=(20, 13), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.35)

    def A(pos, ti, yl):
        a = fig.add_subplot(pos); a.set_facecolor("#1a1a1a")
        a.tick_params(colors='#aaaaaa', labelsize=8); a.spines[:].set_color('#333')
        a.set_title(ti, color='white', fontsize=9.5, pad=7)
        a.set_ylabel(yl, color='#aaaaaa', fontsize=8)
        a.set_xlabel("step t", color='#aaaaaa', fontsize=8); return a

    ax1 = A(gs[0, 0], "Predictive error ||R - R_world||_F", "error (↓ better)")
    for n, (c, ls, lb) in pal.items():
        ax1.plot(x, logs[n]["err"], color=c, lw=2, ls=ls, label=lb, alpha=0.9)
    ax1.legend(fontsize=7, facecolor="#111", labelcolor="white")

    ax2 = A(gs[0, 1], "Spectral richness (entropy Γ)", "entropy")
    for n, (c, ls, _) in pal.items():
        ax2.plot(x, logs[n]["ent"], color=c, lw=2, ls=ls, alpha=0.9)

    ax3 = A(gs[0, 2], "Dynamic stability ||ΔR||_F", "Frobenius")
    for n, (c, ls, _) in pal.items():
        ax3.plot(x, logs[n]["frob"], color=c, lw=2, ls=ls, alpha=0.9)

    ax4 = A(gs[1, 0:2], "Adaptive α_LLM / α_WM evolution", "α value")
    ax4.plot(x, logs["adapt"]["al"], color="#4488ff", lw=2, label="α_LLM")
    ax4.plot(x, logs["adapt"]["aw"], color="#ff6633", lw=2, label="α_WM")
    ax4.axhline(0.5, color="#666", lw=1, ls="--", label="fixed 50/50")
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=7.5, facecolor="#111", labelcolor="white")

    ax5 = A(gs[1, 2], "Π₂ activations (uncertainty > threshold)", "0/1")
    ax5.fill_between(x, 0, logs["adapt"]["pi2"], color="#ffcc00", alpha=0.6,
                     label="Π₂ active")
    ax5.set_ylim(-0.1, 1.3)
    ax5.legend(fontsize=7.5, facecolor="#111", labelcolor="white")

    gain = [logs["fixed"]["err"][t] - logs["adapt"]["err"][t]
            for t in range(steps)]
    ax6 = A(gs[2, 0:2], "Adaptive gain vs fixed alpha (fixed_err - adapt_err)",
            "gain (↑ better)")
    ax6.bar(x, gain, color=["#00ff99" if g > 0 else "#ff4444" for g in gain],
            alpha=0.75, width=0.8)
    ax6.axhline(0, color="white", lw=0.8)
    ax6.text(steps*0.6, max(gain)*0.85 if max(gain) > 0 else 0.1,
             f"Mean gain: {np.mean(gain):+.4f}\n"
             f"Positive steps: {sum(g>0 for g in gain)}/{steps}",
             color="white", fontsize=8.5,
             bbox=dict(facecolor="#222", edgecolor="#555", boxstyle="round"))

    ax7 = A(gs[2, 2], "Phase portrait: α_WM vs error", "predictive error")
    ax7.set_xlabel("α_WM", color='#aaaaaa', fontsize=8)
    sc = ax7.scatter(logs["adapt"]["aw"], logs["adapt"]["err"],
                     c=x, cmap="plasma", s=25, alpha=0.85)
    plt.colorbar(sc, ax=ax7, label="step t").ax.yaxis.label.set_color("white")

    fig.suptitle(
        "Felvia Matrix — Adaptive Alpha V2 (EMA-normalized)\n"
        "© Michael FELVIA 2026 — github.com/YOUR_USERNAME/felvia-matrix",
        color='white', fontsize=11, y=0.995)
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(); print(f"Figure saved: {path}")

def verdict(logs, steps):
    half = steps // 2
    print("\n══════════════════════════════════════════════")
    print("  VERDICT — Adaptive Alpha V2")
    print("══════════════════════════════════════════════")
    def m(k, metric): return np.mean(logs[k][metric][half:])
    labels = {"llm": "LLM only", "wm": "WM only",
              "fixed": "Felvia fixed", "adapt": "Felvia ADAPTIVE ★"}
    print(f"\n{'':22} {'Pred err':>12} {'Entropy':>10} {'Stability':>10}")
    print("─" * 58)
    for k, lb in labels.items():
        print(f"  {lb:20} {m(k,'err'):>12.4f} {m(k,'ent'):>10.4f} "
              f"{m(k,'frob'):>10.4f}")
    al_mean = np.mean(logs["adapt"]["al"])
    aw_mean = np.mean(logs["adapt"]["aw"])
    n_pi2   = sum(logs["adapt"]["pi2"])
    gain = [logs["fixed"]["err"][t] - logs["adapt"]["err"][t]
            for t in range(steps)]
    print(f"\n  Mean α_LLM: {al_mean:.3f}  |  Mean α_WM: {aw_mean:.3f}")
    print(f"  Π₂ active : {int(n_pi2)}/{steps} steps ({100*n_pi2/steps:.0f}%)")
    print(f"  Mean gain vs fixed: {np.mean(gain):+.4f}")
    print(f"  Positive steps    : {sum(g>0 for g in gain)}/{steps}")
    c2 = m("adapt", "err") < m("fixed", "err")
    balanced = 0.25 < aw_mean < 0.75
    print(f"\n  Adaptive < fixed error  : {'✅' if c2 else '✗'}")
    print(f"  Alpha balanced [.25-.75]: {'✅' if balanced else '✗'} (α_WM={aw_mean:.3f})")
    print("══════════════════════════════════════════════")

if __name__ == "__main__":
    print("Felvia Matrix — Adaptive Alpha V2 | 60 steps | D=8\n")
    logs, steps = run(steps=60)
    verdict(logs, steps)
    plot(logs, steps)
    print("\nExperiment complete.")
