import matplotlib.pyplot as plt
from PIL import Image
import io

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use serif fonts by default
    "font.serif": ["Computer Modern"],  # Match LaTeX's default font
    "axes.labelsize": 12,  # Set axis label font size
    "font.size": 12,  # Set general font size
    "legend.fontsize": 10,  # Set legend font size
    "xtick.labelsize": 10,  # Set x-axis tick label font size
    "ytick.labelsize": 10,  # Set y-axis tick label font size
})

def plot_values_and_policy(values, policy, title_values="Value Function", title_policy="Policy"):
    """
    Creates a matplotlib figure with two subplots:
        1. A heatmap of 'values'
        2. An arrow plot of 'policy'

    :param values: State values
    :param policy: Policy followed by agent
    :param title_values: Title of State Values subplot, defaults to "Value Function"
    :param title_policy: Title of Policy subplot, defaults to "Policy"
    :return: Complete figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left subplot: heatmap of state values
    ax_vals = axes[0]
    im = ax_vals.imshow(values, cmap='coolwarm', origin='upper')

    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax_vals.text(x, y, f"{values[y,x]:.1f}", ha='center', va='center', color='black', fontsize=8)

    ax_vals.set_title(title_values)
    ax_vals.set_xticks([])
    ax_vals.set_yticks([])
    # fig.colorbar(im, ax=ax_vals, fraction=0.046, pad=0.04)

    # Right subplot: arrows for policy
    ax_pol = axes[1]
    ax_pol.imshow([[0]*values.shape[1]]*values.shape[0], cmap='binary', origin='upper')
    ax_pol.set_title(title_policy)
    ax_pol.set_xticks([])
    ax_pol.set_yticks([])

    # Draw arrows. If multiple actions are optimal, draw multiple arrows.
    # Policy is a 2D array of sets/lists of best actions, or a 2D array of single best actions.
    # Suppose policy[r, c] = [0, 2] means Up, Left are tied for best.
    arrow_dict = {0: (0, -0.4),  # Up
                  1: (0, +0.4),  # Down
                  2: (-0.4, 0), # Left
                  3: (+0.4, 0)} # Right

    for y in range(policy.shape[0]):
        for x in range(policy.shape[1]):
            actions = policy[y, x]
            for a in actions:
                dx, dy = arrow_dict[a]
                # We invert row -> y in the plotting coords, so carefully place arrow
                # If 'r' is down, we do:
                ax_pol.arrow(x, y, dx, dy,
                             head_width=0.1, head_length=0.1,
                             fc='k', ec='k', length_includes_head=True)

    plt.tight_layout()
    return fig


def fig_to_pil(fig):
    """
    Convert a Matplotlib figure to a Pillow Image for GIF generation or saving.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img
