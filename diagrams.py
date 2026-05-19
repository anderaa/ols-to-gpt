"""
Diagrams for the spectrum.ipynb notebook.

Each function renders one figure from the notebook's architecture progression.
Helper functions at the top are reused by multiple step functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# helpers ---------------------------------------------------------------------

def _box(ax, x, y, w, h, color, lines, fontsize=10, alpha=0.88):
    """draw a rounded rectangle with one or more centred text lines."""
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.08', facecolor=color,
                 edgecolor='white', lw=1.5, alpha=alpha, zorder=2))
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for k, line in enumerate(lines):
        offset = (n - 1) / 2 - k
        ax.text(x + w / 2, y + h / 2 + offset * 0.3, line,
                ha='center', va='center', color='white',
                fontsize=fontsize, fontweight='bold', zorder=3)


def _arr(ax, x0, y0, x1, y1, color='gray', lw=1.2, alpha=1.0):
    """draw a straight arrow from (x0, y0) to (x1, y1)."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha),
                zorder=1)


def _plus(ax, x, y, r=0.28, fontsize=16):
    """draw a + circle (used for residual connection addition points)."""
    ax.add_patch(plt.Circle((x, y), r, color='white', zorder=3, ec='gray', lw=2.0))
    ax.text(x, y, '+', ha='center', va='center',
            fontsize=fontsize, color='gray', fontweight='bold', zorder=4)


def draw_network(ax, layer_sizes, activations, title):
    """generic feedforward network diagram with blue/orange/red nodes."""
    n_layers = len(layer_sizes)
    node_radius = 0.18
    h_gap = 2.2
    v_gap = 1.0

    node_positions = []
    for l, n in enumerate(layer_sizes):
        x = l * h_gap
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * v_gap
        node_positions.append([(x, o) for o in offsets])

    for l in range(n_layers - 1):
        for (x0, y0) in node_positions[l]:
            for (x1, y1) in node_positions[l + 1]:
                ax.plot([x0, x1], [y0, y1], color='gray', linewidth=0.8, alpha=0.5, zorder=1)

    def node_color(l, n_layers):
        if l == n_layers - 1:
            return 'firebrick'
        if l == 0:
            return 'steelblue'
        return 'darkorange'

    for l, (positions, act) in enumerate(zip(node_positions, activations)):
        color = node_color(l, n_layers)
        for x, y in positions:
            ax.add_patch(plt.Circle((x, y), node_radius, color=color,
                                    zorder=2, ec='white', lw=1.5))
        if act:
            mid_x = positions[0][0]
            mid_y = min(p[1] for p in positions) - 0.55
            ax.text(mid_x, mid_y, act, ha='center', va='top', fontsize=9,
                    color='dimgray', style='italic')

    for i, (x, y) in enumerate(node_positions[0]):
        ax.text(x - 0.35, y, f'$x_{{{i+1}}}$', ha='right', va='center', fontsize=11)
    ox, oy = node_positions[-1][0]
    ax.text(ox + 0.35, oy, r'$\hat{y}$', ha='left', va='center', fontsize=11)

    for positions in node_positions:
        n = len(positions)
        mid_x = positions[0][0]
        top_y = max(p[1] for p in positions) + 0.45
        ax.text(mid_x, top_y, f'{n} unit{"s" if n > 1 else ""}',
                ha='center', va='bottom', fontsize=8, color='gray')

    max_nodes = max(layer_sizes)
    ax.set_xlim(-0.8, (n_layers - 1) * h_gap + 0.8)
    ax.set_ylim(-max_nodes * v_gap / 2 - 1.0, max_nodes * v_gap / 2 + 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def draw_onehot_network(ax, n_in, n_hid, n_out, out_labels, title):
    """one-hot input feedforward network (no embedding table)."""
    node_r = 0.18
    ax.set_xlim(-1.0, 8.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    in_ys = np.linspace(4.5, -4.5, n_in)
    for y in in_ys:
        ax.add_patch(plt.Circle((0.0, y), node_r, color='steelblue',
                                zorder=2, ec='white', lw=1.5))
    ax.text(0.0, -5.1, f'{n_in} units', ha='center', fontsize=8, color='gray')
    ax.text(0.0, 5.0, 'one-hot input', ha='center', fontsize=8,
            color='dimgray', style='italic')

    hid_ys = np.linspace(3.5, -3.5, n_hid)
    for y in hid_ys:
        ax.add_patch(plt.Circle((2.8, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        for iy in in_ys:
            ax.plot([0.0 + node_r, 2.8 - node_r], [iy, y],
                    color='gray', lw=0.4, alpha=0.2, zorder=1)
    ax.text(2.8, -4.1, f'{n_hid} units', ha='center', fontsize=8, color='gray')
    ax.text(2.8, 4.1, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    out_ys = np.linspace(2.0, -2.0, n_out)
    for k, y in enumerate(out_ys):
        ax.add_patch(plt.Circle((5.6, y), node_r, color='firebrick',
                                zorder=2, ec='white', lw=1.5))
        for hy in hid_ys:
            ax.plot([2.8 + node_r, 5.6 - node_r], [hy, y],
                    color='gray', lw=0.4, alpha=0.2, zorder=1)
        ax.text(5.6 + 0.35, y, out_labels[k], ha='left', va='center', fontsize=9)
    ax.text(5.6, min(out_ys) - 0.6, 'softmax', ha='center', fontsize=8,
            color='dimgray', style='italic')


def draw_embedding_network(ax, emb_label, n_emb, n_hid, n_out, out_labels, title):
    """embedding lookup feedforward network (with shared E table)."""
    node_r = 0.18
    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    for sym, y_pos in zip(['i₁', 'i₂'], [1.0, -1.0]):
        ax.add_patch(mpatches.FancyBboxPatch((-1.3, y_pos - 0.3), 0.8, 0.6,
                     boxstyle='round,pad=0.05', facecolor='steelblue',
                     edgecolor='white', lw=1.5))
        ax.text(-0.9, y_pos, sym, ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')

    ax.add_patch(mpatches.FancyBboxPatch((-0.1, -1.6), 1.2, 3.2,
                 boxstyle='round,pad=0.05', facecolor='darkorange',
                 alpha=0.85, edgecolor='white', lw=1.5))
    ax.text(0.5, 0, f'E\n({emb_label})', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')
    ax.text(0.5, -2.0, 'shared\nembedding', ha='center', va='center',
            fontsize=8, color='dimgray', style='italic')
    for y_pos in [1.0, -1.0]:
        ax.annotate('', xy=(-0.1, y_pos), xytext=(-0.5, y_pos),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    emb_ys = np.linspace(4.5, -4.5, n_emb)
    for y in emb_ys:
        ax.add_patch(plt.Circle((2.2, y), node_r, color='steelblue',
                                zorder=2, ec='white', lw=1.5))
        ax.annotate('', xy=(2.2 - node_r, y), xytext=(1.1, np.clip(y, -1.4, 1.4)),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6, alpha=0.4))
    ax.text(2.2, -5.1, f'{n_emb} units', ha='center', fontsize=8, color='gray')
    ax.text(2.2, 5.0, 'concat(e₁,e₂)', ha='center', fontsize=8,
            color='dimgray', style='italic')

    hid_ys = np.linspace(3.5, -3.5, n_hid)
    for y in hid_ys:
        ax.add_patch(plt.Circle((4.4, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        for ey in emb_ys:
            ax.plot([2.2 + node_r, 4.4 - node_r], [ey, y],
                    color='gray', lw=0.4, alpha=0.2, zorder=1)
    ax.text(4.4, -4.1, f'{n_hid} units', ha='center', fontsize=8, color='gray')
    ax.text(4.4, 4.1, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    out_ys = np.linspace(2.0, -2.0, n_out)
    for k, y in enumerate(out_ys):
        ax.add_patch(plt.Circle((6.6, y), node_r, color='firebrick',
                                zorder=2, ec='white', lw=1.5))
        for hy in hid_ys:
            ax.plot([4.4 + node_r, 6.6 - node_r], [hy, y],
                    color='gray', lw=0.4, alpha=0.2, zorder=1)
        ax.text(6.6 + 0.35, y, out_labels[k], ha='left', va='center', fontsize=9)
    ax.text(6.6, min(out_ys) - 0.6, 'softmax', ha='center', fontsize=8,
            color='dimgray', style='italic')


# step diagrams ---------------------------------------------------------------

def step6():
    """Architecture comparison: logistic regression vs one hidden layer."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Architecture comparison: logistic regression vs. one hidden layer',
                 fontsize=13, fontweight='bold', y=1.01)
    draw_network(axes[0], layer_sizes=[2, 1], activations=['', 'sigmoid'],
                 title='Step 5 — Logistic regression\nLinear(2→1) + Sigmoid')
    draw_network(axes[1], layer_sizes=[2, 4, 1], activations=['', 'sigmoid', 'sigmoid'],
                 title='Step 6 — Neural network\nLinear(2→4) + Sigmoid + Linear(4→1) + Sigmoid')
    plt.tight_layout()
    plt.show()


def step7():
    """Architecture comparison: sigmoid vs ReLU hidden activation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Architecture comparison: sigmoid vs. ReLU hidden activation',
                 fontsize=13, fontweight='bold', y=1.01)
    draw_network(axes[0], layer_sizes=[2, 4, 1], activations=['', 'sigmoid', 'sigmoid'],
                 title='Step 6 — Sigmoid hidden layer\nLinear(2→4) + Sigmoid + Linear(4→1) + Sigmoid')
    draw_network(axes[1], layer_sizes=[2, 4, 1], activations=['', 'ReLU', 'sigmoid'],
                 title='Step 7 — ReLU hidden layer\nLinear(2→4) + ReLU + Linear(4→1) + Sigmoid')
    plt.tight_layout()
    plt.show()


def step7_activations():
    """Sigmoid vs ReLU activation function and gradient plots."""
    z = np.linspace(-5, 5, 300)
    sig = 1 / (1 + np.exp(-z))
    sig_grad = sig * (1 - sig)
    relu = np.maximum(0, z)
    relu_grad = (z > 0).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Sigmoid vs. ReLU', fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(z, sig, color='steelblue', linewidth=2, label='sigmoid')
    ax.plot(z, relu, color='firebrick', linewidth=2, label='ReLU')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('z')
    ax.set_title('Activation function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z, sig_grad, color='steelblue', linewidth=2, label="sigmoid'")
    ax.plot(z, relu_grad, color='firebrick', linewidth=2, label="ReLU'")
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('z')
    ax.set_title('Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def step8():
    """Architecture comparison: binary vs multi-class output."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Architecture comparison: binary vs. multi-class output',
                 fontsize=13, fontweight='bold', y=1.01)
    draw_network(axes[0], layer_sizes=[2, 4, 1], activations=['', 'ReLU', 'sigmoid'],
                 title='Step 7 — Binary output\nLinear(2→4) + ReLU + Linear(4→1) + Sigmoid')
    draw_network(axes[1], layer_sizes=[2, 4, 5], activations=['', 'ReLU', 'softmax'],
                 title='Step 8 — Multi-class output\nLinear(2→4) + ReLU + Linear(4→5) + Softmax')
    plt.tight_layout()
    plt.show()


def step9():
    """Architecture comparison: continuous vs one-hot inputs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Architecture comparison: continuous vs. one-hot inputs',
                 fontsize=13, fontweight='bold', y=1.01)
    draw_network(axes[0], layer_sizes=[2, 4, 5], activations=['', 'ReLU', 'softmax'],
                 title='Step 8 — Continuous inputs\n2 continuous features → Linear(2→4)')
    draw_network(axes[1], layer_sizes=[10, 8, 5], activations=['', 'ReLU', 'softmax'],
                 title='Step 9 — One-hot inputs\n2×5 one-hot vectors → Linear(10→8)')
    plt.tight_layout()
    plt.show()


def step10(out_vocab=('a', 'b', 'c', 'd', 'e')):
    """Architecture comparison: one-hot vs embedding inputs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Architecture comparison: one-hot encoding vs. learned embeddings',
                 fontsize=13, fontweight='bold', y=1.01)
    prob_labels = [f'P({c})' for c in out_vocab]
    draw_onehot_network(axes[0], n_in=10, n_hid=8, n_out=5,
                        out_labels=prob_labels,
                        title='Step 9 — One-hot inputs\n2×5 one-hot → Linear(10→8)')
    draw_embedding_network(axes[1], emb_label='5×4', n_emb=8, n_hid=8, n_out=5,
                           out_labels=prob_labels,
                           title='Step 10 — Learned embeddings\nEmbed(5×4) + concat → Linear(8→8)')
    plt.tight_layout()
    plt.show()


def step11():
    """Character-level LM architecture (same as step 10, larger vocab)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Step 11 — same architecture as step 10, larger vocabulary',
                 fontsize=12, fontweight='bold')

    node_r = 0.18
    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('2-char context → embedding → hidden → next char',
                 fontsize=12, fontweight='bold', pad=10)

    for sym, y_pos in zip(['i₁', 'i₂'], [1.0, -1.0]):
        ax.add_patch(mpatches.FancyBboxPatch((-1.3, y_pos - 0.3), 0.8, 0.6,
                     boxstyle='round,pad=0.05', facecolor='steelblue',
                     edgecolor='white', lw=1.5))
        ax.text(-0.9, y_pos, sym, ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')

    ax.add_patch(mpatches.FancyBboxPatch((-0.3, -1.6), 1.8, 3.2,
                 boxstyle='round,pad=0.05', facecolor='darkorange',
                 alpha=0.85, edgecolor='white', lw=1.5))
    ax.text(0.6, 0, 'E\n(|vocab|×4)', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')
    ax.text(0.6, -2.0, 'shared\nembedding', ha='center', va='center',
            fontsize=8, color='dimgray', style='italic')
    for y_pos in [1.0, -1.0]:
        ax.annotate('', xy=(-0.3, y_pos), xytext=(-0.5, y_pos),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    emb_ys = np.linspace(4.5, -4.5, 10)
    for y in emb_ys:
        ax.add_patch(plt.Circle((2.2, y), node_r, color='steelblue',
                                zorder=2, ec='white', lw=1.5))
        ax.annotate('', xy=(2.2 - node_r, y), xytext=(1.1, np.clip(y, -1.4, 1.4)),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6, alpha=0.4))
    ax.text(2.2, -5.1, '10 units', ha='center', fontsize=8, color='gray')
    ax.text(2.2, 5.0, 'concat(e₁,e₂)', ha='center', fontsize=8,
            color='dimgray', style='italic')

    hid_ys = np.linspace(3.5, -3.5, 8)
    for y in hid_ys:
        ax.add_patch(plt.Circle((4.4, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        for ey in emb_ys:
            ax.plot([2.2 + node_r, 4.4 - node_r], [ey, y],
                    color='gray', lw=0.4, alpha=0.2, zorder=1)
    ax.text(4.4, -4.1, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(4.4, 4.1, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    for hy in hid_ys:
        ax.plot([4.4 + node_r, 6.3], [hy, 0],
                color='gray', lw=0.4, alpha=0.2, zorder=1)
    ax.add_patch(mpatches.FancyBboxPatch((6.3, -1.5), 2.2, 3.0,
                 boxstyle='round,pad=0.05', facecolor='firebrick',
                 edgecolor='white', lw=1.5))
    ax.text(7.4, 0, 'prob for\neach char\nin vocab', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')
    ax.text(7.4, -2.1, 'softmax', ha='center', fontsize=8,
            color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()


def step12(out_vocab=('a', 'b', 'c', 'd', 'e')):
    """Step 12 architecture: attention replaces concatenation."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Step 12 — attention replaces concatenation, network unchanged',
                 fontsize=12, fontweight='bold')

    node_r = 0.18
    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Embed → Attention → concat → Linear(8→8) → Linear(8→|V|)',
                 fontsize=11, fontweight='bold', pad=10)

    for sym, y_pos in zip(['i₁', 'i₂'], [1.0, -1.0]):
        ax.add_patch(mpatches.FancyBboxPatch((-1.3, y_pos - 0.3), 0.8, 0.6,
                     boxstyle='round,pad=0.05', facecolor='steelblue',
                     edgecolor='white', lw=1.5))
        ax.text(-0.9, y_pos, sym, ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')

    ax.add_patch(mpatches.FancyBboxPatch((-0.3, -1.6), 1.8, 3.2,
                 boxstyle='round,pad=0.05', facecolor='darkorange',
                 alpha=0.85, edgecolor='white', lw=1.5))
    ax.text(0.6, 0, 'E\n(|V|×4)', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')
    for y_pos in [1.0, -1.0]:
        ax.annotate('', xy=(-0.3, y_pos), xytext=(-0.5, y_pos),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    # green attention box
    ax.add_patch(mpatches.FancyBboxPatch((1.9, -1.8), 1.6, 3.6,
                 boxstyle='round,pad=0.05', facecolor='seagreen',
                 alpha=0.85, edgecolor='white', lw=1.5))
    ax.text(2.7, 0.2, 'Self-\nAttention', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')
    ax.text(2.7, -1.2, 'Q,K,V→O', ha='center', va='center',
            color='white', fontsize=9)
    ax.annotate('', xy=(1.9, 0.5), xytext=(1.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('', xy=(1.9, -0.5), xytext=(1.5, -0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    emb_ys = np.linspace(3.5, -3.5, 8)
    for y in emb_ys:
        ax.add_patch(plt.Circle((4.1, y), node_r, color='steelblue',
                                zorder=2, ec='white', lw=1.5))
        ax.annotate('', xy=(4.1 - node_r, y), xytext=(3.5, np.clip(y, -1.6, 1.6)),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6, alpha=0.4))
    ax.text(4.1, -4.1, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(4.1, 4.1, 'concat(o₁,o₂)', ha='center', fontsize=8,
            color='dimgray', style='italic')

    hid_ys = np.linspace(3.5, -3.5, 8)
    for y in hid_ys:
        ax.add_patch(plt.Circle((6.1, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        for ey in emb_ys:
            ax.plot([4.1 + node_r, 6.1 - node_r], [ey, y],
                    color='gray', lw=0.4, alpha=0.2, zorder=1)
    ax.text(6.1, -4.1, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(6.1, 4.1, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    for hy in hid_ys:
        ax.plot([6.1 + node_r, 7.8], [hy, 0],
                color='gray', lw=0.4, alpha=0.2, zorder=1)
    ax.add_patch(mpatches.FancyBboxPatch((7.8, -1.5), 2.2, 3.0,
                 boxstyle='round,pad=0.05', facecolor='firebrick',
                 edgecolor='white', lw=1.5))
    ax.text(8.9, 0, 'prob for\neach char\nin vocab', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')
    ax.text(8.9, -2.1, 'softmax', ha='center', fontsize=8,
            color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()


def _draw_lm_arch(ax, attn_box_label, attn_box_sublabel, title):
    """shared layout: tokens → embeddings → +-circle → attention block → last output → hidden → classifier.

    used by steps 14-17 (and similar) where only the green attention box content changes.
    """
    node_r = 0.18
    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    for sym, y_pos in zip(['i₁', 'i₂'], [1.0, -1.0]):
        ax.add_patch(mpatches.FancyBboxPatch((-1.3, y_pos - 0.3), 0.8, 0.6,
                     boxstyle='round,pad=0.05', facecolor='steelblue',
                     edgecolor='white', lw=1.5))
        ax.text(-0.9, y_pos, sym, ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')

    ax.add_patch(mpatches.FancyBboxPatch((-0.3, -1.6), 1.8, 3.2,
                 boxstyle='round,pad=0.05', facecolor='darkorange',
                 alpha=0.85, edgecolor='white', lw=1.5))
    ax.text(0.6, 0, 'E\n(|V|×4)', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')


def step13():
    """Step 13: causal masking — same as step 12 but Self-Attention → Causal Attention."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Step 13 — causal masking added, everything else unchanged',
                 fontsize=12, fontweight='bold')

    node_r = 0.18
    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Embed → Causal Attention → concat → Linear(8→8) → Linear(8→|V|)',
                 fontsize=11, fontweight='bold', pad=10)

    for sym, y_pos in zip(['i₁', 'i₂'], [1.0, -1.0]):
        ax.add_patch(mpatches.FancyBboxPatch((-1.3, y_pos - 0.3), 0.8, 0.6,
                     boxstyle='round,pad=0.05', facecolor='steelblue',
                     edgecolor='white', lw=1.5))
        ax.text(-0.9, y_pos, sym, ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')

    ax.add_patch(mpatches.FancyBboxPatch((-0.3, -1.6), 1.8, 3.2,
                 boxstyle='round,pad=0.05', facecolor='darkorange',
                 alpha=0.85, edgecolor='white', lw=1.5))
    ax.text(0.6, 0, 'E\n(|V|×4)', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')
    for y_pos in [1.0, -1.0]:
        ax.annotate('', xy=(-0.3, y_pos), xytext=(-0.5, y_pos),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    ax.add_patch(mpatches.FancyBboxPatch((1.9, -1.8), 1.6, 3.6,
                 boxstyle='round,pad=0.05', facecolor='seagreen',
                 alpha=0.85, edgecolor='white', lw=1.5))
    ax.text(2.7, 0.2, 'Causal\nAttention', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')
    ax.text(2.7, -1.2, 'Q,K,V→O', ha='center', va='center',
            color='white', fontsize=9)
    ax.annotate('', xy=(1.9, 0.5), xytext=(1.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('', xy=(1.9, -0.5), xytext=(1.5, -0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    emb_ys = np.linspace(3.5, -3.5, 8)
    for y in emb_ys:
        ax.add_patch(plt.Circle((4.1, y), node_r, color='steelblue',
                                zorder=2, ec='white', lw=1.5))
        ax.annotate('', xy=(4.1 - node_r, y), xytext=(3.5, np.clip(y, -1.6, 1.6)),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6, alpha=0.4))
    ax.text(4.1, -4.1, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(4.1, 4.1, 'concat(o₁,o₂)', ha='center', fontsize=8,
            color='dimgray', style='italic')

    hid_ys = np.linspace(3.5, -3.5, 8)
    for y in hid_ys:
        ax.add_patch(plt.Circle((6.1, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        for ey in emb_ys:
            ax.plot([4.1 + node_r, 6.1 - node_r], [ey, y],
                    color='gray', lw=0.4, alpha=0.2, zorder=1)
    ax.text(6.1, -4.1, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(6.1, 4.1, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    for hy in hid_ys:
        ax.plot([6.1 + node_r, 7.8], [hy, 0],
                color='gray', lw=0.4, alpha=0.2, zorder=1)
    ax.add_patch(mpatches.FancyBboxPatch((7.8, -1.5), 2.2, 3.0,
                 boxstyle='round,pad=0.05', facecolor='firebrick',
                 edgecolor='white', lw=1.5))
    ax.text(8.9, 0, 'prob for\neach char\nin vocab', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')
    ax.text(8.9, -2.1, 'softmax', ha='center', fontsize=8,
            color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()


def step14():
    """Step 14: 5-char context, last output only feeds an 8-unit hidden layer."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Step 14 — causal attention, last output only → Linear(4→8) → Linear(8→|V|)',
                 fontsize=11, fontweight='bold', pad=10)

    def arr(x0, y0, x1, y1, color='gray', lw=1.0, alpha=1.0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha),
                    zorder=1)

    tok_ys = np.linspace(5.8, 1.2, 5)
    x_tok, x_emb, x_attn, x_out, x_hid, x_cls = 0.2, 1.4, 3.0, 4.8, 6.2, 7.8

    for k, y in enumerate(tok_ys):
        _box(ax, x_tok, y - 0.25, 0.8, 0.5, 'steelblue', f'i{k+1}', fontsize=10)
        arr(x_tok + 0.8, y, x_emb, y)

    _box(ax, x_emb, 1.0, 1.2, 5.2, 'darkorange', ['E', '(|V|x4)'], fontsize=11)
    for y in tok_ys:
        arr(x_emb + 1.2, y, x_attn, y)

    _box(ax, x_attn, 1.0, 1.4, 5.2, 'seagreen', ['Causal', 'Attention', 'Q,K,V+mask'], fontsize=10)
    for y in tok_ys:
        arr(x_attn + 1.4, y, x_out, y)

    for k, y in enumerate(tok_ys):
        is_last = (k == len(tok_ys) - 1)
        color = 'steelblue' if is_last else '#aac4e0'
        alpha = 1.0 if is_last else 0.5
        ax.add_patch(plt.Circle((x_out + 0.2, y), 0.18, color=color,
                                zorder=2, ec='white', lw=1.5, alpha=alpha))

    ax.text(x_out + 0.2, 0.7, 'o5 only', ha='center', fontsize=8,
            color='seagreen', style='italic')

    hid_ys = np.linspace(5.5, 1.5, 8)
    for hy in hid_ys:
        ax.plot([x_out + 0.38, x_hid + 0.02], [tok_ys[-1], hy],
                color='gray', lw=0.4, alpha=0.3, zorder=1)

    for y in hid_ys:
        ax.add_patch(plt.Circle((x_hid + 0.2, y), 0.18, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        ax.plot([x_hid + 0.38, x_cls], [y, 3.5],
                color='gray', lw=0.4, alpha=0.3, zorder=1)
    ax.text(x_hid + 0.2, 0.9, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(x_hid + 0.2, 6.3, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    _box(ax, x_cls, 1.8, 2.0, 3.4, 'firebrick',
         ['prob for', 'each char', 'in vocab'], fontsize=10)
    ax.text(x_cls + 1.0, 1.4, 'Linear(8→|V|) + softmax',
            ha='center', fontsize=8, color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()


def step15():
    """Step 15: positional embeddings added before attention."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Step 15 — token + positional embeddings summed before attention',
                 fontsize=11, fontweight='bold', pad=10)

    def arr(x0, y0, x1, y1, color='gray', lw=1.0, alpha=1.0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha),
                    zorder=1)

    x_tok, x_emb, x_plus, x_attn, x_out, x_hid, x_cls = 0.2, 1.6, 3.5, 4.5, 6.3, 7.5, 8.9
    EMB_W, EMB_H = 1.4, 2.5
    attn_ys = np.linspace(5.6, 1.4, 5)

    for k, y in enumerate(attn_ys):
        _box(ax, x_tok, y - 0.25, 0.8, 0.5, 'steelblue', f'i{k+1}', fontsize=10)

    _box(ax, x_emb, 4.2, EMB_W, EMB_H, 'darkorange', ['E', '(|V|x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 4.0, 'token emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    _box(ax, x_emb, 0.8, EMB_W, EMB_H, '#7b5ea7', ['P', '(5x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 0.6, 'pos emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        ax.plot([x_tok + 0.8, x_emb], [y, 5.45], color='gray', lw=0.7, alpha=0.5, zorder=1)
        ax.plot([x_tok + 0.8, x_emb], [y, 2.05], color='gray', lw=0.7, alpha=0.5, zorder=1)

    arr(x_emb + EMB_W, 5.45, x_plus, 3.5, color='darkorange', lw=1.2)
    arr(x_emb + EMB_W, 2.05, x_plus, 3.5, color='#7b5ea7', lw=1.2)
    _plus(ax, x_plus, 3.5)

    for y in attn_ys:
        ax.plot([x_plus + 0.28, x_attn], [3.5, y], color='gray', lw=0.8, alpha=0.6, zorder=1)

    _box(ax, x_attn, 1.0, 1.4, 5.2, 'seagreen', ['Causal', 'Attention'], fontsize=10)
    for y in attn_ys:
        arr(x_attn + 1.4, y, x_out, y, alpha=0.7)

    node_r = 0.18
    for k, y in enumerate(attn_ys):
        is_last = (k == len(attn_ys) - 1)
        ax.add_patch(plt.Circle((x_out + 0.2, y), node_r,
                                color='steelblue' if is_last else '#aac4e0',
                                zorder=2, ec='white', lw=1.5,
                                alpha=1.0 if is_last else 0.4))
    ax.text(x_out + 0.2, 0.7, 'o5 only', ha='center', fontsize=8,
            color='seagreen', style='italic')

    hid_ys = np.linspace(5.5, 1.5, 8)
    for hy in hid_ys:
        ax.plot([x_out + 0.38, x_hid + 0.02], [attn_ys[-1], hy],
                color='gray', lw=0.4, alpha=0.3, zorder=1)

    for y in hid_ys:
        ax.add_patch(plt.Circle((x_hid + 0.2, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        ax.plot([x_hid + 0.38, x_cls], [y, 3.5],
                color='gray', lw=0.4, alpha=0.3, zorder=1)
    ax.text(x_hid + 0.2, 0.9, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(x_hid + 0.2, 6.3, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    _box(ax, x_cls, 1.8, 1.9, 3.4, 'firebrick',
         ['prob for', 'each char', 'in vocab'], fontsize=9)
    ax.text(x_cls + 0.95, 1.4, 'Linear(8→|V|) + softmax',
            ha='center', fontsize=7.5, color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()


def step16():
    """Step 16: single attention box split into two parallel heads."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Step 16 — single attention box split into two parallel heads',
                 fontsize=11, fontweight='bold', pad=10)

    def arr(x0, y0, x1, y1, color='gray', lw=1.0, alpha=1.0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha),
                    zorder=1)

    x_tok, x_emb, x_plus, x_attn, x_cat, x_out, x_hid, x_cls = 0.2, 1.6, 3.5, 4.5, 6.3, 7.3, 8.5, 9.9
    EMB_W, EMB_H = 1.4, 2.5
    attn_ys = np.linspace(5.6, 1.4, 5)

    for k, y in enumerate(attn_ys):
        _box(ax, x_tok, y - 0.25, 0.8, 0.5, 'steelblue', f'i{k+1}', fontsize=10)

    _box(ax, x_emb, 4.2, EMB_W, EMB_H, 'darkorange', ['E', '(|V|x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 4.0, 'token emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')
    _box(ax, x_emb, 0.8, EMB_W, EMB_H, '#7b5ea7', ['P', '(5x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 0.6, 'pos emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        ax.plot([x_tok + 0.8, x_emb], [y, 5.45], color='gray', lw=0.7, alpha=0.5, zorder=1)
        ax.plot([x_tok + 0.8, x_emb], [y, 2.05], color='gray', lw=0.7, alpha=0.5, zorder=1)

    arr(x_emb + EMB_W, 5.45, x_plus, 3.5, color='darkorange', lw=1.2)
    arr(x_emb + EMB_W, 2.05, x_plus, 3.5, color='#7b5ea7', lw=1.2)
    _plus(ax, x_plus, 3.5)

    y_h1, y_h2 = 4.8, 2.2
    ax.plot([x_plus + 0.28, x_attn], [3.5, y_h1], color='gray', lw=0.9, alpha=0.7, zorder=1)
    ax.plot([x_plus + 0.28, x_attn], [3.5, y_h2], color='gray', lw=0.9, alpha=0.7, zorder=1)

    _box(ax, x_attn, y_h1 - 0.7, 1.4, 1.4, 'seagreen', ['Attn', 'Head 1'], fontsize=9)
    _box(ax, x_attn, y_h2 - 0.7, 1.4, 1.4, '#2e8b57', ['Attn', 'Head 2'], fontsize=9)

    arr(x_attn + 1.4, y_h1, x_cat, 4.2, color='seagreen', lw=1.1)
    arr(x_attn + 1.4, y_h2, x_cat, 2.8, color='#2e8b57', lw=1.1)

    _box(ax, x_cat, 2.2, 0.8, 2.6, '#3a6a5a', ['cat', '5x4'], fontsize=9)
    ax.text(x_cat + 0.4, 1.95, 'concat', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        ax.plot([x_cat + 0.8, x_out], [3.5, y], color='gray', lw=0.5, alpha=0.4, zorder=1)

    node_r = 0.18
    for k, y in enumerate(attn_ys):
        is_last = (k == len(attn_ys) - 1)
        ax.add_patch(plt.Circle((x_out + 0.2, y), node_r,
                                color='steelblue' if is_last else '#aac4e0',
                                zorder=2, ec='white', lw=1.5,
                                alpha=1.0 if is_last else 0.4))
    ax.text(x_out + 0.2, 0.7, 'o5 only', ha='center', fontsize=8,
            color='seagreen', style='italic')

    hid_ys = np.linspace(5.5, 1.5, 8)
    for hy in hid_ys:
        ax.plot([x_out + 0.38, x_hid + 0.02], [attn_ys[-1], hy],
                color='gray', lw=0.4, alpha=0.3, zorder=1)

    for y in hid_ys:
        ax.add_patch(plt.Circle((x_hid + 0.2, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        ax.plot([x_hid + 0.38, x_cls], [y, 3.5],
                color='gray', lw=0.4, alpha=0.3, zorder=1)
    ax.text(x_hid + 0.2, 0.9, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(x_hid + 0.2, 6.3, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    _box(ax, x_cls, 1.8, 1.9, 3.4, 'firebrick',
         ['prob for', 'each char', 'in vocab'], fontsize=9)
    ax.text(x_cls + 0.95, 1.4, 'Linear(8→|V|) + softmax',
            ha='center', fontsize=7.5, color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()


def step17_block():
    """Step 17: standalone transformer block diagram (MHA + residual + LayerNorm)."""
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('The transformer block', fontsize=12, fontweight='bold', pad=10)

    def arr(x0, y0, x1, y1, color='gray', lw=1.2):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)

    cx = 2.5
    rx = 4.5

    ax.text(cx, 7.55, 'x', ha='center', fontsize=13, color='steelblue', fontweight='bold')
    ax.text(cx + 0.25, 7.55, '(T × d_e)', ha='left', fontsize=9, color='dimgray')
    arr(cx, 7.3, cx, 6.65)

    ax.plot([cx, rx], [7.3, 7.3], color='steelblue', lw=1.5, zorder=1)
    ax.plot([rx, rx], [7.3, 4.4], color='steelblue', lw=1.5, zorder=1)
    ax.text(rx + 0.15, 5.8, 'x  (residual)', ha='left', fontsize=8,
            color='steelblue', style='italic')

    _box(ax, 1.0, 5.5, 3.0, 1.0, 'seagreen', ['Multi-Head', 'Attention'], fontsize=10)
    arr(cx, 5.5, cx, 4.72)

    _plus(ax, cx, 4.4)
    arr(rx, 4.4, cx + 0.28, 4.4, color='steelblue', lw=1.5)
    arr(cx, 4.12, cx, 3.45)

    _box(ax, 1.0, 2.4, 3.0, 0.9, '#7b5ea7', 'LayerNorm', fontsize=10)
    arr(cx, 2.4, cx, 1.7)

    ax.text(cx, 1.4, "x'", ha='center', fontsize=13, color='steelblue', fontweight='bold')
    ax.text(cx + 0.25, 1.4, '(T × d_e)', ha='left', fontsize=9, color='dimgray')

    ax.add_patch(mpatches.FancyBboxPatch((0.3, 1.8), 4.8, 5.2,
                 boxstyle='round,pad=0.1', facecolor='none',
                 edgecolor='gray', lw=1.5, linestyle='--', zorder=0))
    ax.text(0.55, 6.88, 'transformer block', fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    plt.show()


def step17_arch():
    """Step 17: full architecture with attention wrapped in a transformer block."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Step 17 — attention heads wrapped in a transformer block',
                 fontsize=11, fontweight='bold', pad=10)

    def arr(x0, y0, x1, y1, color='gray', lw=1.0, alpha=1.0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha),
                    zorder=1)

    x_tok, x_emb, x_plus, x_attn, x_out, x_hid, x_cls = 0.2, 1.6, 3.5, 4.6, 7.2, 8.4, 9.8
    EMB_W, EMB_H = 1.4, 2.5
    attn_ys = np.linspace(5.6, 1.4, 5)

    for k, y in enumerate(attn_ys):
        _box(ax, x_tok, y - 0.25, 0.8, 0.5, 'steelblue', f'i{k+1}', fontsize=10)

    _box(ax, x_emb, 4.2, EMB_W, EMB_H, 'darkorange', ['E', '(|V|x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 4.0, 'token emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')
    _box(ax, x_emb, 0.8, EMB_W, EMB_H, '#7b5ea7', ['P', '(5x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 0.6, 'pos emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        ax.plot([x_tok + 0.8, x_emb], [y, 5.45], color='gray', lw=0.7, alpha=0.5, zorder=1)
        ax.plot([x_tok + 0.8, x_emb], [y, 2.05], color='gray', lw=0.7, alpha=0.5, zorder=1)

    arr(x_emb + EMB_W, 5.45, x_plus, 3.5, color='darkorange', lw=1.2)
    arr(x_emb + EMB_W, 2.05, x_plus, 3.5, color='#7b5ea7', lw=1.2)
    _plus(ax, x_plus, 3.5)

    for y in attn_ys:
        ax.plot([x_plus + 0.28, x_attn], [3.5, y], color='gray', lw=0.8, alpha=0.5, zorder=1)

    _box(ax, x_attn, 1.0, 2.2, 5.2, '#00897b', ['Transformer', 'Block'], fontsize=11)
    for y in attn_ys:
        arr(x_attn + 2.2, 3.5, x_out, y, alpha=0.5)

    node_r = 0.18
    for k, y in enumerate(attn_ys):
        is_last = (k == len(attn_ys) - 1)
        ax.add_patch(plt.Circle((x_out + 0.2, y), node_r,
                                color='steelblue' if is_last else '#aac4e0',
                                zorder=2, ec='white', lw=1.5,
                                alpha=1.0 if is_last else 0.4))
    ax.text(x_out + 0.2, 0.7, 'o5 only', ha='center', fontsize=8,
            color='seagreen', style='italic')

    hid_ys = np.linspace(5.5, 1.5, 8)
    for hy in hid_ys:
        ax.plot([x_out + 0.38, x_hid + 0.02], [attn_ys[-1], hy],
                color='gray', lw=0.4, alpha=0.3, zorder=1)

    for y in hid_ys:
        ax.add_patch(plt.Circle((x_hid + 0.2, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        ax.plot([x_hid + 0.38, x_cls], [y, 3.5],
                color='gray', lw=0.4, alpha=0.3, zorder=1)
    ax.text(x_hid + 0.2, 0.9, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(x_hid + 0.2, 6.3, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    _box(ax, x_cls, 1.8, 1.9, 3.4, 'firebrick',
         ['prob for', 'each char', 'in vocab'], fontsize=9)
    ax.text(x_cls + 0.95, 1.4, 'Linear(8→|V|) + softmax',
            ha='center', fontsize=7.5, color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()


def step18():
    """Step 18: stacked transformer blocks (small GPT)."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Step 18 — stacked transformer blocks (small GPT)',
                 fontsize=11, fontweight='bold', pad=10)

    def arr(x0, y0, x1, y1, color='gray', lw=1.0, alpha=1.0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha),
                    zorder=1)

    x_tok, x_emb, x_plus, x_blk1, x_blk2, x_out, x_hid, x_cls = (
        0.2, 1.6, 3.5, 4.6, 7.2, 9.4, 10.6, 11.8)
    EMB_W, EMB_H = 1.4, 2.5
    attn_ys = np.linspace(5.6, 1.4, 5)

    for k, y in enumerate(attn_ys):
        _box(ax, x_tok, y - 0.25, 0.8, 0.5, 'steelblue', f'i{k+1}', fontsize=10)

    _box(ax, x_emb, 4.2, EMB_W, EMB_H, 'darkorange', ['E', '(|V|x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 4.0, 'token emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')
    _box(ax, x_emb, 0.8, EMB_W, EMB_H, '#7b5ea7', ['P', '(5x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 0.6, 'pos emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        ax.plot([x_tok + 0.8, x_emb], [y, 5.45], color='gray', lw=0.7, alpha=0.5, zorder=1)
        ax.plot([x_tok + 0.8, x_emb], [y, 2.05], color='gray', lw=0.7, alpha=0.5, zorder=1)

    arr(x_emb + EMB_W, 5.45, x_plus, 3.5, color='darkorange', lw=1.2)
    arr(x_emb + EMB_W, 2.05, x_plus, 3.5, color='#7b5ea7', lw=1.2)
    _plus(ax, x_plus, 3.5)

    for y in attn_ys:
        ax.plot([x_plus + 0.28, x_blk1], [3.5, y], color='gray', lw=0.7, alpha=0.5, zorder=1)

    _box(ax, x_blk1, 1.0, 1.8, 5.2, '#00897b', ['Transformer', 'Block 1'], fontsize=10)
    ax.text(x_blk1 + 0.9, 0.7, 'T × d_e', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        arr(x_blk1 + 1.8, y, x_blk2, y, alpha=0.5)
    ax.text((x_blk1 + 1.8 + x_blk2) / 2, 6.2, 'full T×d_e', ha='center',
            fontsize=8, color='dimgray', style='italic')

    _box(ax, x_blk2, 1.0, 1.8, 5.2, '#00695c', ['Transformer', 'Block 2'], fontsize=10)
    ax.text(x_blk2 + 0.9, 0.7, 'T × d_e', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        ax.plot([x_blk2 + 1.8, x_out], [y, attn_ys[-1]], color='gray', lw=0.4, alpha=0.35, zorder=1)

    node_r = 0.18
    for k, y in enumerate(attn_ys):
        is_last = (k == len(attn_ys) - 1)
        ax.add_patch(plt.Circle((x_out + 0.2, y), node_r,
                                color='steelblue' if is_last else '#aac4e0',
                                zorder=2, ec='white', lw=1.5,
                                alpha=1.0 if is_last else 0.4))
    ax.text(x_out + 0.2, 0.7, 'o5 only', ha='center', fontsize=8,
            color='seagreen', style='italic')

    hid_ys = np.linspace(5.5, 1.5, 8)
    for hy in hid_ys:
        ax.plot([x_out + 0.38, x_hid + 0.02], [attn_ys[-1], hy],
                color='gray', lw=0.4, alpha=0.3, zorder=1)

    for y in hid_ys:
        ax.add_patch(plt.Circle((x_hid + 0.2, y), node_r, color='darkorange',
                                zorder=2, ec='white', lw=1.5))
        ax.plot([x_hid + 0.38, x_cls], [y, 3.5],
                color='gray', lw=0.4, alpha=0.3, zorder=1)
    ax.text(x_hid + 0.2, 0.9, '8 units', ha='center', fontsize=8, color='gray')
    ax.text(x_hid + 0.2, 6.3, 'ReLU', ha='center', fontsize=8,
            color='dimgray', style='italic')

    _box(ax, x_cls, 1.8, 1.5, 3.4, 'firebrick',
         ['prob for', 'each char', 'in vocab'], fontsize=9)
    ax.text(x_cls + 0.75, 1.4, 'Linear + softmax',
            ha='center', fontsize=7.5, color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()


def step19_block():
    """Step 19: transformer block comparison — attention only vs. with FFN sublayer."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 10))
    fig.suptitle('Transformer block: step 18 vs. step 19',
                 fontsize=12, fontweight='bold', y=1.01)

    def draw_block(ax, title, has_ffn):
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.5, 10.5)
        ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

        def arr(x0, y0, x1, y1, color='gray', lw=1.2):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)

        def bypass(branch_y, plus_y, label_y=None):
            ax.plot([cx, rx], [branch_y, branch_y], color='steelblue', lw=1.4, zorder=1)
            ax.plot([rx, rx], [branch_y, plus_y], color='steelblue', lw=1.4, zorder=1)
            arr(rx, plus_y, cx + 0.25, plus_y, color='steelblue', lw=1.4)
            if label_y is not None:
                ax.text(rx + 0.15, label_y, 'x', ha='left', fontsize=8,
                        color='steelblue', style='italic')

        cx, rx = 2.5, 4.1
        ax.text(cx, 9.7, 'x  (T × d_e)', ha='center', fontsize=9,
                color='dimgray', fontweight='bold')

        if has_ffn:
            arr(cx, 9.4, cx, 8.7)
            _box(ax, 1.2, 7.85, 2.6, 0.85, 'seagreen', ['Multi-Head', 'Attention'], fontsize=9)
            arr(cx, 7.85, cx, 7.1)
            _plus(ax, cx, 6.8, r=0.25, fontsize=14)
            bypass(branch_y=9.4, plus_y=6.8, label_y=8.0)
            arr(cx, 6.55, cx, 5.85)
            _box(ax, 1.2, 5.0, 2.6, 0.85, '#7b5ea7', 'LayerNorm 1', fontsize=9)
            arr(cx, 5.0, cx, 4.3)
            _box(ax, 1.2, 3.45, 2.6, 0.85, '#e07b39', ['FFN', 'GELU  4xd_e'], fontsize=9)
            arr(cx, 3.45, cx, 2.75)
            _plus(ax, cx, 2.45, r=0.25, fontsize=14)
            bypass(branch_y=4.65, plus_y=2.45, label_y=3.7)
            arr(cx, 2.2, cx, 1.45)
            _box(ax, 1.2, 0.6, 2.6, 0.85, '#7b5ea7', 'LayerNorm 2', fontsize=9)
            arr(cx, 0.6, cx, 0.05)
            ax.text(cx, -0.2, "x'  (T × d_e)", ha='center', fontsize=9,
                    color='dimgray', fontweight='bold')
            ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.4), 4.4, 8.6,
                         boxstyle='round,pad=0.1', facecolor='none',
                         edgecolor='gray', lw=1.5, linestyle='--', zorder=0))
        else:
            arr(cx, 9.4, cx, 8.05)
            _box(ax, 1.2, 7.2, 2.6, 0.85, 'seagreen', ['Multi-Head', 'Attention'], fontsize=9)
            arr(cx, 7.2, cx, 6.45)
            _plus(ax, cx, 6.15, r=0.25, fontsize=14)
            bypass(branch_y=9.4, plus_y=6.15, label_y=7.8)
            arr(cx, 5.9, cx, 5.2)
            _box(ax, 1.2, 4.35, 2.6, 0.85, '#7b5ea7', 'LayerNorm', fontsize=9)
            arr(cx, 4.35, cx, 3.7)
            ax.text(cx, 3.4, "x'  (T × d_e)", ha='center', fontsize=9,
                    color='dimgray', fontweight='bold')
            ax.add_patch(mpatches.FancyBboxPatch((0.3, 3.0), 4.4, 6.0,
                         boxstyle='round,pad=0.1', facecolor='none',
                         edgecolor='gray', lw=1.5, linestyle='--', zorder=0))

    draw_block(axes[0], 'Step 18 — post-norm (attention only)', has_ffn=False)
    draw_block(axes[1], 'Step 19 — post-norm + FFN sublayer', has_ffn=True)

    plt.tight_layout()
    plt.show()


def step19_arch():
    """Step 19: full architecture, last block output goes straight to vocabulary projection."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Step 19 — full architecture: last block output goes directly to vocabulary projection',
                 fontsize=11, fontweight='bold', pad=10)

    def arr(x0, y0, x1, y1, color='gray', lw=1.0, alpha=1.0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha),
                    zorder=1)

    x_tok, x_emb, x_plus, x_blk1, x_blk2, x_out, x_cls = 0.2, 1.6, 3.5, 4.6, 7.2, 9.4, 10.8
    EMB_W, EMB_H = 1.4, 2.5
    attn_ys = np.linspace(5.6, 1.4, 5)

    for k, y in enumerate(attn_ys):
        _box(ax, x_tok, y - 0.25, 0.8, 0.5, 'steelblue', f'i{k+1}', fontsize=10)

    _box(ax, x_emb, 4.2, EMB_W, EMB_H, 'darkorange', ['E', '(|V|x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 4.0, 'token emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')
    _box(ax, x_emb, 0.8, EMB_W, EMB_H, '#7b5ea7', ['P', '(5x4)'], fontsize=10)
    ax.text(x_emb + EMB_W/2, 0.6, 'pos emb', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        ax.plot([x_tok + 0.8, x_emb], [y, 5.45], color='gray', lw=0.7, alpha=0.5, zorder=1)
        ax.plot([x_tok + 0.8, x_emb], [y, 2.05], color='gray', lw=0.7, alpha=0.5, zorder=1)

    arr(x_emb + EMB_W, 5.45, x_plus, 3.5, color='darkorange', lw=1.2)
    arr(x_emb + EMB_W, 2.05, x_plus, 3.5, color='#7b5ea7', lw=1.2)
    _plus(ax, x_plus, 3.5)

    for y in attn_ys:
        ax.plot([x_plus + 0.28, x_blk1], [3.5, y], color='gray', lw=0.7, alpha=0.5, zorder=1)

    _box(ax, x_blk1, 1.0, 1.8, 5.2, '#00897b', ['Transformer', 'Block 1'], fontsize=10)
    ax.text(x_blk1 + 0.9, 0.7, 'T × d_e', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        arr(x_blk1 + 1.8, y, x_blk2, y, alpha=0.5)
    ax.text((x_blk1 + 1.8 + x_blk2) / 2, 6.2, 'full T×d_e', ha='center',
            fontsize=8, color='dimgray', style='italic')

    _box(ax, x_blk2, 1.0, 1.8, 5.2, '#00695c', ['Transformer', 'Block 2'], fontsize=10)
    ax.text(x_blk2 + 0.9, 0.7, 'T × d_e', ha='center', fontsize=7.5,
            color='dimgray', style='italic')

    for y in attn_ys:
        ax.plot([x_blk2 + 1.8, x_out], [y, attn_ys[-1]], color='gray', lw=0.4, alpha=0.35, zorder=1)

    node_r = 0.18
    for k, y in enumerate(attn_ys):
        is_last = (k == len(attn_ys) - 1)
        ax.add_patch(plt.Circle((x_out + 0.2, y), node_r,
                                color='steelblue' if is_last else '#aac4e0',
                                zorder=2, ec='white', lw=1.5,
                                alpha=1.0 if is_last else 0.4))
    ax.text(x_out + 0.2, 0.7, 'o_T only', ha='center', fontsize=8,
            color='seagreen', style='italic')

    arr(x_out + 0.38, attn_ys[-1], x_cls, 3.5, color='steelblue', lw=1.5)

    _box(ax, x_cls, 1.8, 2.4, 3.4, 'firebrick',
         ['prob for', 'each char', 'in vocab'], fontsize=9)
    ax.text(x_cls + 1.2, 1.4, 'Linear(d_e → |V|) + softmax',
            ha='center', fontsize=7.5, color='dimgray', style='italic')

    plt.tight_layout()
    plt.show()
