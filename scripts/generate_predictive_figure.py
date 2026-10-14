"""Known-spectrum NMR validation: joint spectra lead, diagnostics support.

Truth and posterior mean use the same density scale. Pointwise width and the
T2 marginal interval quantify spectral uncertainty; coverage summarizes all cases.
Native spectral cells are drawn without smoothing or spatial interpolation.
"""
from experiment_data import *
from matplotlib.colors import Normalize
from matplotlib.ticker import NullLocator

rid = 'SYN_S01_R01'
spectrum = sorted(read('spectral_map.csv'),
                  key=lambda r: (int(r['t1_index']), int(r['t2_index'])))
edge1 = np.r_[sorted({float(r['log10_t1_left_s']) for r in spectrum}),
              max(values(spectrum, 'log10_t1_right_s'))]
edge2 = np.r_[sorted({float(r['log10_t2_left_s']) for r in spectrum}),
              max(values(spectrum, 'log10_t2_right_s'))]
truth, mean, width = [values(spectrum, field).reshape(32, 32)
                      for field in ['truth_density', 'mean_density', 'width90_density']]
norm = Normalize(vmin=0, vmax=max(truth.max(), mean.max()))
fig = plt.figure(figsize=(6.9, 4.1))

def area(x, y, w, h):
    return fig.add_axes([x/6.9, y/4.1, w/6.9, h/4.1])

for j, (z, letter, label) in enumerate(zip(
        [truth, mean, width], 'abc', ['True density', 'Mean density', '90% width'])):
    x = [.46, 2.68, 4.90][j]
    ax = area(x, 2.06, 1.69, 1.50)
    mesh = ax.pcolormesh(10**(edge2+3), 10**(edge1+3), z,
                         cmap=DENSITY_CMAP if j < 2 else UNCERTAINTY_CMAP,
                         norm=norm if j < 2 else Normalize(0, width.max()), shading='flat')
    ax.set(xscale='log', yscale='log', xlim=(.1, 1e4), ylim=(.1, 1e4),
           xlabel=r'$T_2$ (ms)', ylabel=r'$T_1$ (ms)' if j == 0 else '',
           xticks=[.1, 10, 1000, 10000], yticks=[.1, 10, 1000, 10000])
    ax.tick_params(axis='y', labelleft=j == 0)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    panel(ax, letter)
    cax = area(x, 3.79, 1.69, .055)
    fig.colorbar(mesh, cax=cax, orientation='horizontal', label=label)
    cax.xaxis.set_label_position('top')
    cax.tick_params(pad=1, length=2)

ax = area(.46, .46, 3.74, .85)
marginal = sorted([r for r in read('spectral_marginals.csv') if r['record_id'] == rid],
                  key=lambda r: int(r['t2_index']))
edges = 10**(np.r_[values(marginal, 'log10_t2_left_s'),
                    float(marginal[-1]['log10_t2_right_s'])]+3)
lo, hi = values(marginal, 'q05_density'), values(marginal, 'q95_density')
ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step='post',
                color=COLORS[0], alpha=.18, lw=0, label='90% interval')
ax.stairs(values(marginal, 'mean_density'), edges, color=COLORS[0], lw=1.4, label='Mean')
ax.stairs(values(marginal, 'truth_density'), edges, color=COLORS[1], ls='--', lw=1.25, label='Truth')
ax.set(xscale='log', xlabel=r'$T_2$ (ms)', ylabel='Density (a.u.)', xlim=(.1, 1e4),
       ylim=(0, None), xticks=[.1, 1, 10, 100, 1000, 10000])
ax.xaxis.set_minor_locator(NullLocator())
legend_above(ax, ncol=3)
panel(ax, 'd')

ax = area(4.86, .46, 1.76, .85)
score = supplemented('performance_by_unit.csv')
for method, color, marker in zip(METHODS, COLORS, ['o', 's', '^']):
    data = sorted([r for r in score if r['method'] == method and r['level'] == 'macro_site'],
                  key=lambda r: float(r['interval_level']))
    ax.plot(values(data, 'interval_level'), values(data, 'coverage_fraction'),
            marker=marker, color=color, label=LABELS[method], markersize=4)
ax.plot([.45, 1], [.45, 1], color='0.5', ls='--', lw=.8)
ax.set(xlim=(.45, 1), ylim=(.43, 1), xlabel='Nominal probability', ylabel='Coverage',
       xticks=[.5, .65, .8, .95], xticklabels=['.50', '.65', '.80', '.95'])
legend_above(ax, ncol=3)
panel(ax, 'e')
save(fig, 'Fig3_Predictive_Spectral')
