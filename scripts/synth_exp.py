"""
Synthetic experiment for Flat Bundle Neural Networks (Section 4 of the notes).

Seven baselines across four sample-complexity regimes T/Vn ∈ {0.5, 1.5, 5, 15}.

Baselines (from Section 4 of the notes)
----------------------------------------
  SampleVNN   – VNN on the Vn×Vn sample covariance (Sihag et al.)
  DISJ-GL     – n=1, GL graph at preprocessing, frozen; only h trained
  DISJ-KRON   – n=2, O=I, GL graph at preprocessing, frozen; only h trained
  DISJ-FB     – n=2, full flat bundle at preprocessing (learn L and O), frozen
  J-GL        – n=1, jointly learn scalar graph + filter taps
  J-KRON      – n=2, O=I, jointly learn scalar graph + filter taps
  J-FB        – n=2, jointly learn flat bundle (L and O) + filter taps  ← proposed

DISJ variants use a double loop:
  Phase 1 — pretrain_graph() zeros the neural contribution (α=0) and runs
             only the graph-learning alternating steps to convergence.
  Phase 2 — freeze the learned graph, train only the SPNN filter taps (h).

Generative model
----------------
  V=30, n=2, k-NN RBF graph (k=5), SO(2) random frames per node.
  x ~ N(0, L†),   L = O₀ᵀ(L₀⊗I₂)O₀
  y = (1/Vn) 1ᵀ O₀ᵀ(UF(Λ)Uᵀ⊗I₂)O₀ x + noise,  F(Λ)=k1−k2(Λ−λ1)²(Λ−λ2)²
  (eq. 33, λ1=3rd / λ2=12th non-zero eigenvalue of L₀, SNR=10 dB)

Metrics
-------
  All baselines : MAE, MSE
  DISJ-KRON, DISJ-FB, J-KRON, J-FB : additionally F1 (edge-support recovery)

Usage (from repo root):
  conda run -n PNN python scripts/synth_exp.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.generate import (
    generate_sheaf_task,
    project_to_scalar,
    complete_graph_edge_index,
)
from src.training.train_sheaf_pnn import train_sheaf_pnn, evaluate_sheaf_pnn
from src.training.utils import compute_sample_covariance, pretrain_graph
from src.operators.laplacian import build_laplacian, build_blockdiag

# =============================================================================
# Configuration
# =============================================================================
V      = 30
N      = 2           # stalk dimension for Kron / FB variants
K_NN   = 5           # k-NN for RBF graph
T_TEST = 300         # fixed held-out test size
N_SEEDS = 3
SNR_dB  = 10.0

VN = V * N
REGIMES = {
    "Scarce (0.5)": int(0.5  * VN),
    "Low (1.5)":    int(1.5  * VN),
    "Medium (5)":   int(5.0  * VN),
    "High (15)":    int(15.0 * VN),
}

# Shared SPNN arch (same for all variants)
MODEL_KW = dict(
    hidden_sizes=[16, 16],
    hidden_mlp_sizes=[16],
    K=3,
    bias=True,
    dropout=0.0,
    task_level="graph",
    node_readout="mean",
    task="regression",
    use_batch_norms=True,
)

# Neural training hyperparams (Phase 2 / joint)
TRAIN_KW = dict(
    nEpochs=50,
    it_h=15,
    it_z=8,
    lr=0.01,
    eta_z=3e-4,
    alpha=0.5,
    gamma1=1.0,
    gamma2=1.0,
    gamma3=100.0,
    gamma4=100.0,
    lam=0.7,
    epsilon=1e-3,
    penalty="l1",
    verbose=False,
)

# Graph pretraining hyperparams (Phase 1 / DISJ)
PRETRAIN_KW = dict(
    nEpochs=40,
    it_z=8,
    alpha=0.0,
    gamma1=1.0,
    gamma2=1.0,
    gamma3=1.0,
    gamma4=1.0,
    lam=0.05,
    epsilon=1e-3,
    eta_z=3e-4,
    eta_O=0.01,
    penalty="l1",
)

# Variants that report F1 (edge-support recovery)
F1_VARIANTS = {"DISJ-KRON", "DISJ-FB", "J-KRON", "J-FB"}


# =============================================================================
# F1 helper
# =============================================================================

def edge_f1(w_hat, true_edge_index, all_edge_index):
    """Top-k_true threshold F1 for edge-support recovery."""
    k_true = true_edge_index.shape[1]
    true_set = {
        (max(int(true_edge_index[0, k]), int(true_edge_index[1, k])),
         min(int(true_edge_index[0, k]), int(true_edge_index[1, k])))
        for k in range(k_true)
    }
    w_vals = w_hat.detach().cpu()
    top_idx = w_vals.topk(min(k_true, len(w_vals))).indices.tolist()
    est_set = {
        (max(int(all_edge_index[0, k]), int(all_edge_index[1, k])),
         min(int(all_edge_index[0, k]), int(all_edge_index[1, k])))
        for k in top_idx
    }
    TP = len(true_set & est_set)
    if TP == 0:
        return 0.0
    p = TP / len(est_set)
    r = TP / len(true_set)
    return 2 * p * r / (p + r)


def laplacian_to_weights(L, edge_index):
    """Extract (negated off-diagonal) weights from a V×V Laplacian."""
    return (-L[edge_index[0], edge_index[1]]).clamp(min=0.0)


# =============================================================================
# Baseline implementations
# =============================================================================

def run_sample_vnn(X_tr, y_tr, X_te, y_te, seed):
    """SampleVNN: VNN on Vn×Vn sample covariance."""
    Vn, T_tr, _ = X_tr.shape
    C = compute_sample_covariance(X_tr)
    ei = complete_graph_edge_index(Vn)
    w_dummy = torch.ones(ei.shape[1])
    I_bl = torch.eye(1).unsqueeze(0).expand(Vn, -1, -1).clone()
    model, *_ = train_sheaf_pnn(
        X_tr, y_tr, ei, Vn, 1,
        prec_type="fixed", w_init=w_dummy, O_blocks_init=I_bl, L_fixed=C,
        seed=seed, **MODEL_KW, **TRAIN_KW,
    )
    res = evaluate_sheaf_pnn(model, X_te, y_te, w_dummy, I_bl, ei, Vn, 1, L_fixed=C)
    return res, None


def run_disj_gl(X_tr, y_tr, X_te, y_te, graph, seed):
    """DISJ-GL: n=1, GL graph at preprocessing, then freeze; only h trained."""
    V_g = V
    ei_all = complete_graph_edge_index(V_g)

    # Phase 1: learn scalar graph (n=1, O=I)
    X_tr_s = project_to_scalar(X_tr, V_g, N)   # (V, T, 1)
    C_1d = compute_sample_covariance(X_tr_s)    # (V, V)
    w_pre, O_pre = pretrain_graph(
        C_1d, ei_all, V_g, 1, learn_O=False, device="cpu", **PRETRAIN_KW
    )
    L_pre = build_laplacian(w_pre, ei_all, V_g)
    I_bl1 = torch.eye(1).unsqueeze(0).expand(V_g, -1, -1).clone()

    # Phase 2: neural training only with frozen graph
    X_te_s = project_to_scalar(X_te, V_g, N)
    model, *_ = train_sheaf_pnn(
        X_tr_s, y_tr, ei_all, V_g, 1,
        prec_type="fixed", w_init=w_pre, O_blocks_init=I_bl1, L_fixed=L_pre,
        seed=seed, **MODEL_KW, **TRAIN_KW,
    )
    res = evaluate_sheaf_pnn(model, X_te_s, y_te, w_pre, I_bl1, ei_all, V_g, 1, L_fixed=L_pre)
    return res, None   # no F1 for GL variant


def run_disj_kron(X_tr, y_tr, X_te, y_te, graph, seed):
    """DISJ-KRON: n=2, O=I, GL graph at preprocessing, then freeze."""
    V_g = V
    ei_all = complete_graph_edge_index(V_g)
    ei_true = graph["edge_index"]
    I_bl2 = torch.eye(N).unsqueeze(0).expand(V_g, -1, -1).clone()

    # Phase 1: learn graph with Kron structure (O=I)
    C_nd = compute_sample_covariance(X_tr)   # (Vn, Vn)
    w_pre, O_pre = pretrain_graph(
        C_nd, ei_all, V_g, N, learn_O=False, device="cpu", **PRETRAIN_KW
    )
    L_pre = build_laplacian(w_pre, ei_all, V_g)

    # Phase 2: neural training only
    model, *_ = train_sheaf_pnn(
        X_tr, y_tr, ei_all, V_g, N,
        prec_type="fixed", w_init=w_pre, O_blocks_init=I_bl2, L_fixed=L_pre,
        seed=seed, **MODEL_KW, **TRAIN_KW,
    )
    res = evaluate_sheaf_pnn(model, X_te, y_te, w_pre, I_bl2, ei_all, V_g, N, L_fixed=L_pre)
    f1 = edge_f1(w_pre, ei_true, ei_all)
    return res, f1


def run_disj_fb(X_tr, y_tr, X_te, y_te, graph, seed):
    """DISJ-FB: n=2, full flat bundle at preprocessing (learn L+O), then freeze."""
    V_g = V
    ei_all = complete_graph_edge_index(V_g)
    ei_true = graph["edge_index"]

    # Phase 1: learn full flat bundle (L and O), no task loss
    C_nd = compute_sample_covariance(X_tr)
    w_pre, O_pre = pretrain_graph(
        C_nd, ei_all, V_g, N, learn_O=True, device="cpu", **PRETRAIN_KW
    )
    L_pre = build_laplacian(w_pre, ei_all, V_g)

    # Phase 2: neural training only
    model, *_ = train_sheaf_pnn(
        X_tr, y_tr, ei_all, V_g, N,
        prec_type="fixed", w_init=w_pre, O_blocks_init=O_pre,
        seed=seed, **MODEL_KW, **TRAIN_KW,
    )
    res = evaluate_sheaf_pnn(model, X_te, y_te, w_pre, O_pre, ei_all, V_g, N)
    f1 = edge_f1(w_pre, ei_true, ei_all)
    return res, f1


def run_j_gl(X_tr, y_tr, X_te, y_te, graph, seed):
    """J-GL: n=1, jointly learn scalar graph + filter taps."""
    V_g = V
    ei_all = complete_graph_edge_index(V_g)
    I_bl1 = torch.eye(1).unsqueeze(0).expand(V_g, -1, -1).clone()

    X_tr_s = project_to_scalar(X_tr, V_g, N)
    X_te_s = project_to_scalar(X_te, V_g, N)

    model, w_j, O_j, *_ = train_sheaf_pnn(
        X_tr_s, y_tr, ei_all, V_g, 1,
        prec_type="joint", learn_O=False, O_blocks_init=I_bl1,
        seed=seed, **MODEL_KW, **TRAIN_KW,
    )
    res = evaluate_sheaf_pnn(model, X_te_s, y_te, w_j, O_j, ei_all, V_g, 1)
    return res, None


def run_j_kron(X_tr, y_tr, X_te, y_te, graph, seed):
    """J-KRON: n=2, O=I frozen, jointly learn scalar graph + filter taps."""
    V_g = V
    ei_all = complete_graph_edge_index(V_g)
    ei_true = graph["edge_index"]
    I_bl2 = torch.eye(N).unsqueeze(0).expand(V_g, -1, -1).clone()

    model, w_j, O_j, *_ = train_sheaf_pnn(
        X_tr, y_tr, ei_all, V_g, N,
        prec_type="joint", learn_O=False, O_blocks_init=I_bl2,
        seed=seed, **MODEL_KW, **TRAIN_KW,
    )
    res = evaluate_sheaf_pnn(model, X_te, y_te, w_j, O_j, ei_all, V_g, N)
    f1 = edge_f1(w_j, ei_true, ei_all)
    return res, f1


def run_j_fb(X_tr, y_tr, X_te, y_te, graph, seed):
    """J-FB: jointly optimize full flat bundle (L and O) + filter taps — proposed."""
    V_g = V
    ei_all = complete_graph_edge_index(V_g)
    ei_true = graph["edge_index"]

    model, w_j, O_j, *_ = train_sheaf_pnn(
        X_tr, y_tr, ei_all, V_g, N,
        prec_type="joint", learn_O=True,
        seed=seed, **MODEL_KW, **TRAIN_KW,
    )
    res = evaluate_sheaf_pnn(model, X_te, y_te, w_j, O_j, ei_all, V_g, N)
    f1 = edge_f1(w_j, ei_true, ei_all)
    return res, f1


# =============================================================================
# Main experiment loop
# =============================================================================

VARIANTS = [
    ("SampleVNN",  run_sample_vnn,  False),   # (name, fn, needs_graph)
    ("DISJ-GL",    run_disj_gl,     True),
    ("DISJ-KRON",  run_disj_kron,   True),
    ("DISJ-FB",    run_disj_fb,     True),
    ("J-GL",       run_j_gl,        True),
    ("J-KRON",     run_j_kron,      True),
    ("J-FB",       run_j_fb,        True),
]

records = []

for regime_name, T_train in REGIMES.items():
    print(f"\n{'='*70}")
    print(f"Regime: {regime_name}  (T_train={T_train}, Vn={VN})")
    print(f"{'='*70}")

    for seed in range(N_SEEDS):
        data = generate_sheaf_task(
            V=V, n=N, k_nn=K_NN,
            T_train=T_train, T_test=T_TEST,
            seed=seed, SNR_dB=SNR_dB,
        )
        X_tr = data["X_train"]
        y_tr = data["y_train"]
        X_te = data["X_test"]
        y_te = data["y_test"]
        graph = data["graph"]
        lmeta = data["label_meta"]

        # Verify enough non-zero eigenvalues (might fail for very sparse graphs)
        nonzero_eigs = (torch.linalg.eigvalsh(graph["L0"]) > 1e-6).sum().item()
        if nonzero_eigs < 12:
            print(f"  [seed {seed}] SKIP — only {nonzero_eigs} non-zero eigenvalues")
            continue

        for vname, vfunc, needs_graph in VARIANTS:
            if needs_graph:
                res, f1 = vfunc(X_tr, y_tr, X_te, y_te, graph, seed)
            else:
                res, f1 = vfunc(X_tr, y_tr, X_te, y_te, seed)

            mae = res["mae"]
            mse = res["mse"]
            records.append(dict(
                regime=regime_name, seed=seed, variant=vname,
                mae=mae, mse=mse,
                f1=f1 if f1 is not None else float("nan"),
            ))
            f1_str = f"  F1={f1:.3f}" if f1 is not None else ""
            print(f"  [seed {seed}] {vname:<12s}  MAE={mae:.4f}  MSE={mse:.4f}{f1_str}")

# =============================================================================
# Summary table
# =============================================================================
df = pd.DataFrame(records)
os.makedirs("results", exist_ok=True)
df.to_csv("results/synth_exp_raw.csv", index=False)

pivot = df.groupby(["regime", "variant"])[["mae", "mse", "f1"]].agg(["mean", "std"])
print("\n\n" + "="*80)
print("SUMMARY  (mean ± std over seeds)")
print("="*80)
print(pivot.round(4).to_string())

# =============================================================================
# Plots
# =============================================================================
REGIME_ORDER  = list(REGIMES.keys())
VARIANT_ORDER = [v for v, _, _ in VARIANTS]
COLORS  = {
    "SampleVNN": "gray",
    "DISJ-GL":   "lightblue", "DISJ-KRON": "cornflowerblue", "DISJ-FB": "royalblue",
    "J-GL":      "lightsalmon","J-KRON":    "tomato",         "J-FB":    "firebrick",
}
MARKERS = {
    "SampleVNN": "x",
    "DISJ-GL":   "s",  "DISJ-KRON": "^",  "DISJ-FB": "D",
    "J-GL":      "s",  "J-KRON":    "^",  "J-FB":    "o",
}
LINES   = {
    "SampleVNN": ":",
    "DISJ-GL":   "--", "DISJ-KRON": "--", "DISJ-FB": "--",
    "J-GL":      "-",  "J-KRON":    "-",  "J-FB":    "-",
}
x_ticks = list(range(len(REGIME_ORDER)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, metric, ylabel, title in [
    (axes[0], "mae",  "Test MAE",          "Downstream MAE"),
    (axes[1], "mse",  "Test MSE",          "Downstream MSE"),
    (axes[2], "f1",   "F1 (edge support)", "Graph Recovery F1"),
]:
    plot_variants = VARIANT_ORDER if metric != "f1" else [
        v for v in VARIANT_ORDER if v in F1_VARIANTS
    ]
    for vname in plot_variants:
        sub = df[df.variant == vname].groupby("regime")[metric]
        mu  = [sub.mean().get(r, float("nan")) for r in REGIME_ORDER]
        se  = [sub.std().get(r, float("nan")) / max(1, N_SEEDS**0.5) for r in REGIME_ORDER]
        ax.errorbar(
            x_ticks, mu, yerr=se, label=vname,
            color=COLORS[vname], marker=MARKERS[vname],
            linestyle=LINES[vname], linewidth=1.5, capsize=3,
        )
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["Scarce", "Low", "Med", "High"], fontsize=9)
    ax.set_xlabel("Regime  (T/Vn)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2 if metric != "f1" else 1)
    if metric == "f1":
        ax.set_ylim(-0.05, 1.05)

fig.tight_layout()
fig.savefig("results/synth_exp.pdf", bbox_inches="tight")
fig.savefig("results/synth_exp.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/synth_exp_raw.csv, results/synth_exp.{pdf,png}")
