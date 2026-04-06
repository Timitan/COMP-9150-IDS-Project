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
    Grouped bar chart of MCC and Macro-F1 across all four experiments,
    styled after Figure 4 in Cantone et al. (2024).
"""

import numpy as np
import matplotlib.pyplot as plt
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

_EXP_TYPES = {1: 'within-dataset', 2: 'cross-dataset',
              3: 'within-dataset', 4: 'cross-dataset'}


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

    n      = len(labels)
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
        fname = title.replace(' ', '_').replace('/', '-').replace('|', '').strip('_')
        path  = f'{fname}_confusion_matrix.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {path}')

    plt.show()


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary_bar_chart(summary, save: bool = True):
    models  = ['RF', 'XGB']
    n_exp   = 4
    bar_w   = 0.35
    x       = np.arange(n_exp)
    metrics = [('MCC', 'MCC'), ('Macro-F1', 'Macro-F1')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle('MCC & Macro-F1 by Experiment — RF vs XGBoost',
                 fontsize=13, fontweight='bold')

    # Build legend handles manually so all 4 combinations are represented
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=_COLOURS[('within-dataset', 'RF')],  label='RF   — within-dataset'),
        Patch(color=_COLOURS[('within-dataset', 'XGB')], label='XGB — within-dataset'),
        Patch(color=_COLOURS[('cross-dataset',  'RF')],  label='RF   — cross-dataset'),
        Patch(color=_COLOURS[('cross-dataset',  'XGB')], label='XGB — cross-dataset'),
    ]

    for ax, (metric_col, metric_name) in zip(axes, metrics):
        for i, model in enumerate(models):
            vals        = []
            colour_list = []

            for eid in range(1, n_exp + 1):
                row = summary[
                    (summary['Experiment'] == eid) & (summary['Model'] == model)
                ]
                vals.append(row[metric_col].values[0] if len(row) else 0.0)
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

        for xi in [1, 3]:
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.05, color='#4E8AC9')

        ax.axhline(0.5, color='grey', linewidth=0.7, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(_EXP_LABELS, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(handles=legend_handles, fontsize=8, title='Model / Type',
                  loc='upper right')

    plt.tight_layout()

    if save:
        plt.savefig('summary_bar_chart.png', dpi=150, bbox_inches='tight')
        print('  Saved: summary_bar_chart.png')

    plt.show()
