"""Image plate: preserve signed signal structure across depths and receivers.

Use aligned log-coordinate acquisition maps and one fixed scale per receiver.
Export at single-column width with editable text and rasterized measured cells.
"""
from experiment_data import *
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullLocator

fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.05), layout='constrained',
                         sharex=True, sharey=True)
for row, channel in enumerate([1, 2]):
    limit = [75, 38][row]
    for col, record in enumerate(['B10_6-4', 'B10_7-7']):
        data = read(f'{record}_array{channel}_observed.csv', folder='real_data')
        waits = sorted({float(r['waiting_time_ms']) for r in data})
        echoes = sorted({float(r['echo_time_ms']) for r in data})
        lookup = {(float(r['waiting_time_ms']), float(r['echo_time_ms'])):
                  float(r[f'source_signal_array_{channel}']) for r in data}
        z = np.array([[lookup[(w, e)] for e in echoes] for w in waits])
        ax = axes[row, col]
        mesh = ax.pcolormesh(edges_from_centers(echoes, True),
                            edges_from_centers(waits, True), z,
                            cmap=SIGNED_CMAP, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                            shading='flat')
        ax.set(xscale='log', yscale='log', xlabel='Echo time (ms)',
               ylabel='Recovery (ms)' if col == 0 else '', xlim=(.5, 500))
        ax.set_xticks([.5, 10, 500])
        ax.set_xticklabels(['0.5', '10', '500'])
        ax.set_yticks([1, 10, 100, 1000])
        ax.set_yticklabels(['1', '10', '100', '1000'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='x', labelbottom=True)
        panel(ax, 'abcd'[2*row+col])
    colorbar = fig.colorbar(mesh, ax=axes[row,:], shrink=.88, pad=.025,
                           aspect=22, label='Signal', ticks=[-limit, 0, limit])
save(fig, 'Fig2_Public_Borehole_Raw', directory=FIGDIR, svg=True)
