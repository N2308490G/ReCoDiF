"""Compact triptych: regional means conceal distributional shifts.

The profile shows means and central 90% sample ranges. The discrepancy
matrix and empirical CDF distinguish mean from distributional agreement.
"""
from experiment_data import *

rows = read('toy_region_draws.csv')
fields = [f'region_{i:02d}_mass' for i in range(1, 17)]
samples = {g: np.array([[float(r[k]) for k in fields] for r in rows
                       if r['ensemble_id'] == g]) for g in REGION_GROUPS}
energy = np.zeros((4, 4))
for r in read('toy_region_energy.csv'):
    energy[REGION_GROUPS.index(r['ensemble_a']),
           REGION_GROUPS.index(r['ensemble_b'])] = float(r['energy_v'])

fig = plt.figure(figsize=(6.9, 2.15), layout='constrained')
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.05])
ax = fig.add_subplot(gs[0, 0])
for g, label, color, marker, ls in zip(REGION_GROUPS, REGION_LABELS,
        REGIONAL_COLORS, ['o', 's', '^', 'D'], ['-', '-', '--', ':']):
    lo, hi = np.quantile(samples[g], [.05, .95], axis=0)
    x = np.arange(1, 17)
    ax.fill_between(x, lo, hi, color=color, alpha=.10, lw=0)
    ax.plot(x, samples[g].mean(axis=0), color=color, marker=marker,
            markersize=3.3, markeredgewidth=.4, markeredgecolor='white',
            linestyle=ls, label=label)
ax.set(xlabel='Region index', ylabel='Mass (a.u.)',
       xticks=[1, 4, 7, 10, 13, 16], xlim=(.6, 16.4), ylim=(0, None))
legend_above(ax, ncol=4)
panel(ax, 'a')

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(energy*1000, cmap=ERROR_CMAP, vmin=0, vmax=8)
ax.set(xticks=range(4), xticklabels=REGION_LABELS,
       yticks=range(4), yticklabels=REGION_LABELS, xlabel='Distribution')
ax.tick_params(length=0)
for i in range(4):
    for j in range(4):
        value = energy[i, j]*1000
        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                color=cell_text_color(im.cmap, im.norm, value))
fig.colorbar(im, ax=ax, pad=.025, shrink=.85, ticks=[0, 4, 8])
panel(ax, 'b')

ax = fig.add_subplot(gs[0, 2])
for g, label, color, ls in zip(REGION_GROUPS, REGION_LABELS, REGIONAL_COLORS,
                              ['-', '-', '--', ':']):
    x = np.sort(samples[g][:, 9])
    ax.step(x, np.arange(1, len(x)+1)/len(x), where='post',
            label=label, color=color, linestyle=ls, lw=1.35)
ax.set(xlabel='Region-10 mass (a.u.)', ylabel='CDF',
       ylim=(0, 1.02))
legend_above(ax, ncol=4)
panel(ax, 'c')
save(fig, 'Fig4_Regional_Distributions')
