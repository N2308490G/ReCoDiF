"""Resource and convergence composite with three aligned evidence columns.

Repeat-level time/memory distributions describe cost; draw-prefix scores resolve
Monte Carlo behavior; individual optimization paths contextualize the median.
"""
from completed_results import *
from matplotlib.ticker import NullLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.75), layout='constrained')
runtime = rows('runtime_results.csv')
for ax, key, label, letter in [(axes[0, 0], 'total_inference_s', 'CPU time (s)', 'a'),
                               (axes[1, 0], 'peak_memory_mb', 'Peak memory (MiB)', 'd')]:
    medians = []
    ax.axhspan(-.46, .46, color=COLORS[0], alpha=.055, lw=0, zorder=0)
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        data = [r for r in runtime if r['method'] == method]
        samples = values(data, key)
        assert len(samples) == 30 and np.isfinite(samples).all()
        q1, med, q3 = np.quantile(samples, [.25, .5, .75])
        medians.append(med)
        offset = .19*np.sin(np.arange(len(samples))*2.39996323)
        ax.scatter(samples, i+offset, s=7, color=color, alpha=.37, linewidths=0, zorder=2)
        ax.plot([q1, q3], [i, i], color=color, lw=3.3, solid_capstyle='round', zorder=3)
        ax.plot(med, i, 'o', color='white', mec=color, mew=1.1, ms=4.4, zorder=4)
        ax.annotate(f'{med:.1f}', (samples.max(), i), xytext=(5, 0),
                    textcoords='offset points', va='center', color=INK)
    ax.set(xlabel=label, yticks=np.arange(5), yticklabels=[method_label(m) for m in METHODS], ylim=(4.6, -.6))
    ax.tick_params(axis='y', length=0)
    if key == 'total_inference_s':
        ax.set_xscale('log')
        ax.set_xlim(min(medians)*.74, max(medians)*2.7)
        ax.set_xticks([30, 100, 1000], ['30', '100', '1000'])
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_xlim(min(medians)*.72, max(medians)*1.35)
        ax.set_xticks([600, 1000, 1400, 1800])
    panel(ax, letter)

budget = rows('draw_budget_sensitivity.csv')
for ax, key, ylabel, letter in [(axes[0, 1], 'crps', 'Normalized CRPS', 'b'),
                                (axes[1, 1], 'energy_v', 'Regional discrepancy\n(stored units)', 'e')]:
    for record, color, marker, style, label in zip(
            ['B10_6-4', 'B10_7-7'], DEPTH_COLORS, ['o', 's'], ['-', '--'], ['7.0 m', '8.5 m']):
        data = sorted([r for r in budget if r['record_id'] == record
                       and r['method_a'] == 'ReDif'], key=lambda r: number(r['draw_count']))
        assert len(data) == 4
        x, y = values(data, 'draw_count'), values(data, key)
        assert np.isfinite(y).all()
        if key == 'crps':
            ax.errorbar(x, y, yerr=values(data, 'mcse'), color=color, ls=style,
                         marker=marker, ms=3.5, capsize=2, elinewidth=.8, label=label)
        else:
            ax.plot(x, y, color=color, ls=style, marker=marker, ms=3.5, label=label)
    ax.set(xscale='log', xticks=[64, 128, 256, 512],
           xticklabels=['64', '128', '256', '512'], xlabel='Predictive draws', ylabel=ylabel)
    ax.xaxis.set_minor_locator(NullLocator())
    legend_above(ax, ncol=2)
    panel(ax, letter)

curve = [r for r in rows('training_curves.csv') if r['method'] == 'ReDif']
steps = np.array(sorted({number(r['step']) for r in curve}))
run_ids = sorted({r['run_id'] for r in curve})
lookup = {(r['run_id'], number(r['step'])): r for r in curve}
handles = [Line2D([], [], color=COLORS[0], lw=.7, alpha=.45, label='Seeds'),
           Patch(fc=COLORS[0], alpha=.12, label='Range'),
           Line2D([], [], color=COLORS[0], lw=1.6, label='Median')]
for ax, key, letter, ylabel in [(axes[0, 2], 'train_loss', 'c', 'Training objective'),
                                (axes[1, 2], 'validation_metric', 'f', 'Validation objective')]:
    y = np.array([[number(lookup[(run, step)][key]) for run in run_ids] for step in steps])
    assert y.shape == (50, 5)
    x = steps/1000
    ax.fill_between(x, y.min(axis=1), y.max(axis=1), color=COLORS[0], alpha=.10, lw=0)
    for column in y.T:
        ax.plot(x, column, color=COLORS[0], lw=.55, alpha=.30)
    ax.plot(x, np.median(y, axis=1), color=COLORS[0], lw=1.6)
    if key == 'validation_metric':
        for run in run_ids:
            selected = next(r for r in rows('training_runs.csv') if r['run_id'] == run)
            step = number(selected['selected_step'])
            value = number(lookup[(run, step)][key])
            ax.plot(step/1000, value, 'o', ms=3.7, color=DEPTH_COLORS[1],
                    mec='white', mew=.35, zorder=5)
    ax.set(xlabel=r'Updates ($10^3$)', ylabel=ylabel, xlim=(0, 100),
           xticks=[0, 50, 100])
    legend_above(ax, ncol=3, handles=handles, handlelength=.8, columnspacing=.4)
    panel(ax, letter)
save(fig, 'Fig9_Computational_Profiles')
