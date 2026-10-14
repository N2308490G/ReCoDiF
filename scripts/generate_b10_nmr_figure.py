"""NMR image plate: depth-dependent spectra and local method differences.

ReCoDiF mean maps are the primary evidence; differences from CSGM and DAPS
locate distributional changes that scalar prediction scores cannot resolve.
Native, nonuniform 16 x 16 cells are retained without spatial interpolation.
The two rows share mean and difference scales at a 6.9-inch publication width.
"""
from completed_results import *
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.ticker import NullLocator

spectra = b10_spectra()
records = ['B10_6-4', 'B10_7-7']
comparison = ['CSGM', 'DAPS']
means = [spectra[record, 'ReDif']['density'] for record in records]
differences = [[spectra[record, method]['density']-means[i]
                for method in comparison] for i, record in enumerate(records)]
mean_limit = np.ceil(max(z.max() for z in means)/20)*20
difference_limit = np.ceil(max(np.abs(z).max() for row in differences for z in row)/5)*5
mean_norm = PowerNorm(.5, vmin=0, vmax=mean_limit)
diff_norm = TwoSlopeNorm(vmin=-difference_limit, vcenter=0, vmax=difference_limit)

fig = plt.figure(figsize=(6.9, 4.5))
def area(x, y, w, h):
    return fig.add_axes([x/6.9, y/4.5, w/6.9, h/4.5])

columns = [.47, 2.70, 4.93]
for row, (record, depth) in enumerate(zip(records, ['7.0 m', '8.5 m'])):
    data = spectra[record, 'ReDif']
    x, y = 10**(data['edges2']+3), 10**(data['edges1']+3)
    fields = [means[row], *differences[row]]
    for col, z in enumerate(fields):
        ax = area(columns[col], 2.48 if row == 0 else .47, 1.68, 1.48)
        mesh = ax.pcolormesh(x, y, z, shading='flat',
                            cmap=DENSITY_CMAP if col == 0 else SIGNED_CMAP,
                            norm=mean_norm if col == 0 else diff_norm)
        ax.set(xscale='log', yscale='log', xlim=(.1, 1e4), ylim=(.1, 1e4),
               xticks=[.1, 10, 1000, 10000], yticks=[.1, 10, 1000, 10000],
               xlabel=r'$T_2$ (ms)', ylabel=r'$T_1$ (ms)' if col == 0 else '')
        ax.tick_params(axis='y', labelleft=col == 0)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.plot([.1, 1e4], [.1, 1e4], color='#67727E', lw=.55, ls='--', alpha=.65)
        if col == 0:
            ax.text(.035, .96, depth, transform=ax.transAxes, va='top')
        panel(ax, 'abcdef'[3*row+col])
        if row == 0:
            cax = area(columns[col], 4.18, 1.68, .055)
            label = ['ReCoDiF density', 'CSGM − ReCoDiF', 'DAPS − ReCoDiF'][col]
            ticks = [0, 30, 60, 120] if col == 0 else [-difference_limit, 0, difference_limit]
            bar = fig.colorbar(mesh, cax=cax, orientation='horizontal', label=label, ticks=ticks)
            cax.xaxis.set_label_position('top')
            cax.tick_params(pad=1, length=2)
save(fig, 'Fig_B10_NMR_Spectra')
print(f'Common density limit: {mean_limit:g}; difference limit: ±{difference_limit:g}')
