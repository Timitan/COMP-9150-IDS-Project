"""
viz_utils.py
============
Visualisation utilities for the COMP 9150 RF & XGBoost
cross-dataset generalisation experiment.

Functions
---------
plot_confusion_matrix(y_true, y_pred, le, title, save)
    Heatmap confusion matrix for a single experiment / model.

plot_summary_bar_chart(summary, save)
    Grouped bar chart of MCC, Macro-F1, and AUROC across all four
    experiments, styled after Figure 4 in Cantone et al. (2024).

plot_comparison_bar(comparison, metric, title, save)
    Grouped bar chart comparing all-features vs mRMR+undersampling
    runs for a single metric (MCC, Macro-F1, or AUROC).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Colour palette (mirrors Cantone et al. Fig 4 colour scheme)
# ---------------------------------------------------------------------------

_COLOURS = {
    ('within-dataset', 'RF') : '#D7553A',   # red-orange  — within, RF
    ('within-dataset', 'XGB'): '#E8994D',   # amber       — within, XGB
    ('cross-dataset',  'RF') : '#4E8AC9',   # steel blue  — cross,  RF
    ('cross-dataset',  'XGB'): '#5DB87A',   # sage green  — cross,  XGB
}

_EXP_LABELS = [
    'LycoS17\non LycoS17',
    'LycoS17\non LycoS18',
    'LycoS18\non LycoS18',
    'LycoS18\non LycoS17',
]

_EXP_TYPES = {
    1: 'within-dataset',
    2: 'cross-dataset',
    3: 'within-dataset',
    4: 'cross-dataset',
}

_MODEL_COLOURS = {
    'RF' : '#D7553A',
    'XGB': '#4E8AC9',
}


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    le,
    title: str = 'Confusion Matrix',
    save: bool = True,
):
    """
    Plot and optionally save a labelled confusion-matrix heatmap.

    Parameters
    ----------
    y_true : array-like  (integer-encoded ground truth)
    y_pred : array-like  (integer-encoded predictions)
    le     : fitted LabelEncoder  (supplies human-readable class names)
    title  : str  plot title and (if save=True) base filename
    save   : bool  write PNG to disk when True (default True)
    """
    labels = le.classes_
    cm     = confusion_matrix(y_true, y_pred)

    n       = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.4,
        linecolor='#e0e0e0',
    )
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()

    if save:
        fname = (
            title.replace(' ', '_')
                 .replace('/', '-')
                 .replace('|', '')
                 .strip('_')
        )
        path = f'{fname}_confusion_matrix.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {path}')

    plt.show()


# ---------------------------------------------------------------------------
# Summary bar chart  (MCC + Macro-F1 + AUROC)
# ---------------------------------------------------------------------------

def plot_summary_bar_chart(summary, save: bool = True):
    """
    Grouped bar chart of MCC, Macro-F1, and AUROC across the four
    experiments, styled after Cantone et al. (2024) Figure 4.

    Parameters
    ----------
    summary : pd.DataFrame
        Must contain columns: Experiment (int 1-4), Model (str 'RF'/'XGB'),
        Type (str 'within-dataset'/'cross-dataset'), MCC (float),
        Macro-F1 (float), AUROC (float).
    save : bool
        Write PNG to disk when True (default True).
    """
    models  = ['RF', 'XGB']
    n_exp   = 4
    bar_w   = 0.35
    x       = np.arange(n_exp)
    metrics = [('MCC', 'MCC'), ('Macro-F1', 'Macro-F1'), ('AUROC', 'AUROC')]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
    fig.suptitle('MCC, Macro-F1 & AUROC by Experiment — RF vs XGBoost',
                 fontsize=13, fontweight='bold')

    # Build legend handles manually so all 4 type/model combos appear
    legend_handles = [
        mpatches.Patch(color=_COLOURS[('within-dataset', 'RF')],
                       label='RF   — within-dataset'),
        mpatches.Patch(color=_COLOURS[('within-dataset', 'XGB')],
                       label='XGB — within-dataset'),
        mpatches.Patch(color=_COLOURS[('cross-dataset',  'RF')],
                       label='RF   — cross-dataset'),
        mpatches.Patch(color=_COLOURS[('cross-dataset',  'XGB')],
                       label='XGB — cross-dataset'),
    ]

    for ax, (metric_col, metric_name) in zip(axes, metrics):
        # Skip AUROC panel gracefully if column not present
        if metric_col not in summary.columns:
            ax.text(0.5, 0.5, f'{metric_col}\nnot available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='grey')
            ax.set_title(metric_name, fontsize=11)
            ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        for i, model in enumerate(models):
            vals        = []
            colour_list = []

            for eid in range(1, n_exp + 1):
                row = summary[
                    (summary['Experiment'] == eid) &
                    (summary['Model'] == model)
                ]
                val = row[metric_col].values[0] if len(row) else 0.0
                # NaN AUROC (skipped) renders as 0 with a note
                vals.append(0.0 if (isinstance(val, float) and
                                    val != val) else val)
                exp_type = _EXP_TYPES.get(eid, 'within-dataset')
                colour_list.append(_COLOURS[(exp_type, model)])

            offset = (i - (len(models) - 1) / 2) * bar_w
            bars   = ax.bar(
                x + offset, vals, bar_w,
                color=colour_list,
                edgecolor='white',
                linewidth=0.5,
            )

            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7.5,
                )

        # Shade cross-dataset experiment columns
        for xi in [1, 3]:
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.05, color='#4E8AC9')

        ax.axhline(0.5, color='grey', linewidth=0.7, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(_EXP_LABELS, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(handles=legend_handles, fontsize=8,
                  title='Model / Type', loc='upper right')

    plt.tight_layout()

    if save:
        plt.savefig('summary_bar_chart.png', dpi=150, bbox_inches='tight')
        print('  Saved: summary_bar_chart.png')

    plt.show()


# ---------------------------------------------------------------------------
# Comparison bar chart  (all-features vs mRMR + undersampling)
# ---------------------------------------------------------------------------

def plot_comparison_bar(comparison, metric: str, title: str,
                        save: bool = True):
    """
    Grouped bar chart comparing all-features vs mRMR+undersampling runs
    for a single metric (MCC, Macro-F1, or AUROC).

    Solid bars   = all-features baseline (§9).
    Hatched bars = mRMR + undersampling run (§13).

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of the §13.7 merge; must contain columns
        '{metric}_all' and '{metric}_mrmr'.
    metric : str   one of 'MCC', 'Macro-F1', 'AUROC'
    title  : str   plot title
    save   : bool  write PNG to disk when True (default True)
    """
    col_all  = f'{metric}_all'
    col_mrmr = f'{metric}_mrmr'

    if col_all not in comparison.columns or col_mrmr not in comparison.columns:
        print(f'plot_comparison_bar: columns {col_all!r} / {col_mrmr!r} '
              f'not found in comparison DataFrame — skipping.')
        return

    models  = ['RF', 'XGB']
    exp_ids = sorted(comparison['Experiment'].unique())
    n_exp   = len(exp_ids)
    bar_w   = 0.2
    x       = np.arange(n_exp)

    # 4 bar groups: RF-all, RF-mrmr, XGB-all, XGB-mrmr
    offsets = np.linspace(
        -(len(models) - 0.5) * bar_w,
         (len(models) - 0.5) * bar_w,
        len(models) * 2,
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    bar_idx = 0

    for model in models:
        colour = _MODEL_COLOURS[model]

        for col, hatch, run_label in [
            (col_all,  '',    'all features'),
            (col_mrmr, '///', 'mRMR + undersample'),
        ]:
            vals = []
            for eid in exp_ids:
                row = comparison[
                    (comparison['Experiment'] == eid) &
                    (comparison['Model'] == model)
                ]
                val = row[col].values[0] if len(row) else 0.0
                vals.append(0.0 if (isinstance(val, float) and
                                    val != val) else val)

            bars = ax.bar(
                x + offsets[bar_idx], vals, bar_w,
                color=colour,
                hatch=hatch,
                edgecolor='white' if not hatch else colour,
                linewidth=0.5,
                alpha=0.85,
                label=f'{model} — {run_label}',
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7,
                )
            bar_idx += 1

    exp_labels = [
        f'Exp {eid}\n({"within" if eid in (1, 3) else "cross"})'
        for eid in exp_ids
    ]

    # Shade cross-dataset groups
    for xi, eid in enumerate(exp_ids):
        if eid in (2, 4):
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.05, color='#4E8AC9')

    ax.axhline(0.5, color='grey', linewidth=0.7, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, ncol=2, loc='upper right')

    plt.tight_layout()

    if save:
        fname = f'comparison_{metric.lower().replace("-", "_")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')

    plt.show()
