"""
Reusable plot helpers for the spectrum.ipynb notebook.

Each function renders a common visualisation pattern used across multiple
steps: actual-vs-predicted scatter, convergence curves, confusion matrices,
probability distributions, and combined two-panel layouts.
"""

import numpy as np
import matplotlib.pyplot as plt


def actual_vs_predicted(y_test, y_pred, title='Actual vs. predicted — held-out sample'):
    """45-degree scatter of predicted vs. actual values, with MSE in the title."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    mse = float(np.mean((y_test - y_pred) ** 2))

    fig, ax = plt.subplots(figsize=(5, 5))
    lims = [min(y_test.min(), y_pred.min()) - 1,
            max(y_test.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, color='black', linewidth=1.5, linestyle='--',
            label='perfect prediction')
    ax.scatter(y_test, y_pred, color='seagreen', alpha=0.7, s=40)
    ax.set_xlabel('actual y')
    ax.set_ylabel('predicted ŷ')
    ax.set_title(f'{title}\nMSE = {mse:.3f}')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def convergence(loss_hist, title='convergence', ylabel='cross-entropy loss',
                vocab_size=None, y_test=None, y_pred=None):
    """Single-panel convergence curve. Optionally prints test accuracy if y_test, y_pred provided."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_hist, color='steelblue', linewidth=1.5)
    ax.set_xlabel('iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if y_test is not None and y_pred is not None:
        y_test = np.asarray(y_test)
        y_pred = np.asarray(y_pred)
        accuracy = float(np.mean(y_pred == y_test))
        baseline = f'   (random baseline: {1/vocab_size:.3f})' if vocab_size else ''
        print(f'test accuracy: {accuracy:.3f}{baseline}')


def convergence_with_test(loss_hist, test_hist, title='convergence',
                          ylabel='cross-entropy loss'):
    """Mini-batched training: train batch loss + test loss markers on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_hist, color='steelblue', linewidth=1.0, alpha=0.6,
            label='train (batch)')
    if test_hist:
        steps_t, losses_t = zip(*test_hist)
        ax.plot(steps_t, losses_t, color='firebrick', linewidth=2, marker='o',
                markersize=5, label='test')
    ax.set_xlabel('iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def convergence_and_confusion(
    loss_hist,
    y_test,
    y_pred,
    class_labels=None,
    ylabel='log-likelihood',
    conv_title='convergence',
    cm_title='Confusion matrix — held-out sample',
    figsize=None,
):
    """Two-panel: convergence curve on the left, confusion matrix on the right.

    Works for both binary (class_labels=['0', '1']) and multi-class targets.
    If class_labels is None, defaults to the integer indices found in y_test/y_pred.
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    if class_labels is None:
        n_classes = int(max(y_test.max(), y_pred.max())) + 1
        class_labels = [str(i) for i in range(n_classes)]
    else:
        n_classes = len(class_labels)

    # build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[true, pred] += 1
    accuracy = float(np.trace(cm) / cm.sum())

    fs = figsize if figsize else (12 if n_classes <= 3 else 13, 5)
    fig, axes = plt.subplots(1, 2, figsize=fs)

    axes[0].plot(loss_hist, color='steelblue', linewidth=2,
                 marker='o' if len(loss_hist) <= 50 else None,
                 markersize=4)
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(conv_title)
    axes[0].grid(True, alpha=0.3)

    ax = axes[1]
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    fontsize_axis = 11 if n_classes <= 3 else 10
    ax.set_xticklabels([f'pred {c}' for c in class_labels], fontsize=fontsize_axis)
    ax.set_yticklabels([f'actual {c}' for c in class_labels], fontsize=fontsize_axis)
    cell_fs = 18 if n_classes <= 3 else 13
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=cell_fs, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title(f'{cm_title}\naccuracy = {accuracy:.2f}')

    plt.tight_layout()
    plt.show()


def probability_distribution(probs, vocab, seed='', title=None):
    """Bar chart of the model's predicted distribution over a vocabulary.

    Prints the top-5 most likely tokens below the chart.
    """
    probs = np.asarray(probs)
    V = len(probs)

    if title is None:
        title = (f"Model output: P(next char | input = {repr(seed)})"
                 if seed else 'Model output: P(next char)')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(V), probs, color='steelblue', alpha=0.8)
    ax.set_xticks(range(V))
    ax.set_xticklabels([repr(c) for c in vocab], fontsize=9)
    ax.set_xlabel('next character')
    ax.set_ylabel('probability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
    print('Top 5 predicted next characters:')
    for idx, p in top5:
        print(f'  {repr(vocab[idx]):4s}  {p:.3f}')
