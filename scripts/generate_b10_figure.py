"""NMR spectra and signal validation at the same two measured depths.

Method marginals and ReCoDiF posterior intervals lead the composite. Held-out
recovery profiles and residual strips then connect these spectra to observations.
"""
from completed_results import *
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

fig = plt.figure(figsize=(6.9, 4.2), layout='constrained')
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1., .26], hspace=.025)
spectra = b10_spectra()
method_names = METHODS + ['DLEMLR']
method_colors = COLORS + ['#65707B']
line_styles = ['-', '--', '-.', ':', (0, (4, 1)), (0, (1, 1))]
marginal_axes = []
marginal_limits = []
for col, record, letter, depth in zip(range(2), ['B10_6-4', 'B10_7-7'], 'ab', [7., 8.5]):
    ax = fig.add_subplot(gs[0, col])
    marginal_axes.append(ax)
    reference = spectra[record, 'ReDif']
    edges = 10**(reference['edges2']+3)
    lo, hi = reference['q05'], reference['q95']
    ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step='post',
                    color=COLORS[0], alpha=.16, lw=0, label='ReCoDiF 90%')
    for method, color, style in zip(method_names, method_colors, line_styles):
        density = spectra[record, method]['marginal']
        ax.stairs(density, edges, color=color, ls=style,
                  lw=1.45 if method == 'ReDif' else 1.0,
                  zorder=5 if method == 'ReDif' else 3, label=method_label(method))
        marginal_limits.extend([density.min(), density.max()])
    marginal_limits.extend([lo.min(), hi.max()])
    ax.set(xscale='log', xlabel=r'$T_2$ (ms)', ylabel='Density (a.u.)',
           xlim=(.1, 1e4), xticks=[.1, 10, 1000, 10000])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.text(.025, .97, f'{depth:g} m', transform=ax.transAxes, va='top')
    panel(ax, letter)
lo, hi = min(marginal_limits), max(marginal_limits)
for ax in marginal_axes:
    ax.set_ylim(min(lo, 0)-.025*(hi-lo), hi*1.08)
handles, labels = marginal_axes[0].get_legend_handles_labels()
fig.legend(handles[1:]+handles[:1], labels[1:]+labels[:1],
           loc='outside upper center', ncol=7, frameon=False,
           handlelength=1.15, handletextpad=.35, columnspacing=.8, borderpad=.1)

predictions = rows('heldout_predictive_results.csv')
signal_axes, residual_axes = [], []
signal_limits, residual_limits = [], []
for col, record, letter, depth in zip(range(2), ['B10_6-4', 'B10_7-7'], 'cd', [7., 8.5]):
    def interval(probability):
        return sorted([r for r in predictions if r['record_id'] == record
                       and r['method'] == 'ReDif' and r['run_id'] == 'ARM_ReDif_S1001'
                       and r.get('case_id') == 'primary'
                       and number(r['interval_level']) == probability],
                      key=lambda r: int(r['vector_position']))
    outer, inner = interval(.9), interval(.5)
    assert len(outer) == len(inner) == 32
    assert [r['vector_position'] for r in outer] == [r['vector_position'] for r in inner]
    x = np.arange(1, 33)
    mean = values(outer, 'predictive_mean')
    observed = values(outer, 'observed_signal')
    residual = observed-mean
    ax = fig.add_subplot(gs[1, col])
    signal_axes.append(ax)
    for start in [0, 8, 16, 24]:
        segment = slice(start, start+8)
        labels = ['90%', '50%', 'Mean'] if start == 0 else [None]*3
        ax.fill_between(x[segment], values(outer, 'lower')[segment],
                        values(outer, 'upper')[segment], color=COLORS[0],
                        alpha=.13, lw=0, label=labels[0])
        ax.fill_between(x[segment], values(inner, 'lower')[segment],
                        values(inner, 'upper')[segment], color=COLORS[0],
                        alpha=.24, lw=0, label=labels[1])
        ax.plot(x[segment], mean[segment], color=COLORS[0], lw=1.35, label=labels[2])
    ax.scatter(x, observed, s=11, color=INK, edgecolor='white',
               linewidth=.35, zorder=4, label='Observed')
    for edge in [8.5, 16.5, 24.5]:
        ax.axvline(edge, color='#BEC8CE', lw=.55, ls=':')
    ax.set(ylabel='Signal (stored units)', xlim=(.5, 32.5))
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.spines['bottom'].set_visible(False)
    ax.text(.025, .97, f'{depth:g} m', transform=ax.transAxes, va='top')
    legend_above(ax, ncol=4)
    signal_limits.extend([observed.min(), observed.max(),
                          values(outer, 'lower').min(), values(outer, 'upper').max()])
    ax = fig.add_subplot(gs[2, col], sharex=ax)
    residual_axes.append(ax)
    ax.axhline(0, color='#98A7AF', lw=.65)
    ax.vlines(x, 0, residual, color=COLORS[0], lw=.65, alpha=.6)
    ax.scatter(x, residual, color=COLORS[0], s=7, zorder=3)
    ax.set(xlabel='Target position', ylabel='Residual', xticks=[1, 8, 16, 24, 32])
    panel(ax, letter)
    residual_limits.append(abs(residual).max())
lo, hi = min(signal_limits), max(signal_limits)
for ax in signal_axes:
    ax.set_ylim(lo-.05*(hi-lo), hi+.10*(hi-lo))
for ax in residual_axes:
    limit = max(residual_limits)*1.12
    ax.set_ylim(-limit, limit)
    ax.set_yticks([-20, 0, 20])

save(fig, 'Fig7_B10_Prediction')

