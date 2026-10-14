"""Paired quantitative grid: components affect prediction and regional agreement.

Seed-level observations and mean/SD intervals carry the ablation comparison.
The two annotated sensitivity maps retain their distinct numerical scales.
"""
from completed_results import *
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(6.9, 3.75), layout='constrained')
grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
data = variants()
variant_colors = [COLORS[0], COLORS[1], COLORS[1], COLORS[1],
                  COLORS[2], COLORS[2], COLORS[2]]
legend = [Line2D([], [], marker='o', ms=3, ls='', color='#8B9AA4', label='Seed'),
          Line2D([], [], marker='D', ms=3.5, lw=1.5, color=INK, label='Mean ± SD')]
for ax, key, letter in [(axes[0], 'crps', 'a'), (axes[2], 'regional_energy_v', 'c')]:
    full_mean = values(next(iter(data.values())), key).mean()
    ax.axhspan(-.48, .48, color=COLORS[0], alpha=.06, lw=0, zorder=0)
    ax.axvline(full_mean, color=COLORS[0], alpha=.55, lw=.8, ls='--', zorder=0)
    for i, ((variant, subset), color) in enumerate(zip(data.items(), variant_colors)):
        subset = sorted(subset, key=lambda r: number(r['train_seed']))
        x = values(subset, key)
        mean, sd = x.mean(), x.std(ddof=1)
        ax.errorbar(mean, i, xerr=sd, fmt='D', color=color, ms=3.9,
                    capsize=2.5, elinewidth=1.8, markeredgecolor='white',
                    markeredgewidth=.4, zorder=4)
        ax.scatter(x, i+np.linspace(-.17, .17, len(x)), s=13, color=color,
                   alpha=.42, edgecolor='white', linewidth=.4, zorder=3)
    ax.set(yticks=np.arange(7), yticklabels=list(VARIANTS.values()), ylim=(6.6, -.6))
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Normalized CRPS' if key == 'crps' else 'Regional discrepancy (stored units)')
    if key == 'regional_energy_v':
        ax.set_xscale('log')
    legend_above(ax, ncol=2, handles=legend)
    panel(ax, letter)

prior = rows('prior_and_domain_sensitivity.csv')[:9]
for ax, key, letter, cmap in [(axes[1], 'crps', 'b', ERROR_CMAP),
                               (axes[3], 'regional_energy_v', 'd', ERROR_CMAP)]:
    matrix = np.array([number(r[key]) for r in prior]).reshape(3, 3)
    im = ax.imshow(matrix, origin='lower', cmap=cmap, aspect='auto')
    ax.set(xticks=np.arange(3), xticklabels=['0.05', '0.10', '0.20'],
           yticks=np.arange(3), yticklabels=['0.5', '1', '2'],
           xlabel=r'Correlation length $L_1=L_2$', ylabel=r'$a_z$')
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                    color=cell_text_color(im.cmap, im.norm, matrix[i,j]))
    fig.colorbar(im, ax=ax, fraction=.04, pad=.025, shrink=.85)
    panel(ax, letter)
save(fig, 'Fig8_Ablation_Sensitivity')

