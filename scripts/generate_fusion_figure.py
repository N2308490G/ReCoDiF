"""Posterior geometry and intervention response form a quantitative composite.

Grid-based highest-density contours reveal contraction after complementary
observations. Matched response maps show how timing and sample count interact.
"""
from experiment_data import *
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

grid = read('toy_fusion_grid.csv')
m1 = np.sort(np.unique(values(grid, 'mass1')))
m2 = np.sort(np.unique(values(grid, 'mass2')))
cell_area = float(grid[0]['cell_area'])
def matrix(field):
    out = np.zeros((len(m1), len(m2)))
    for r in grid:
        i, j = np.searchsorted(m1, float(r['mass1'])), np.searchsorted(m2, float(r['mass2']))
        out[i, j] = float(r[field])/float(r['cell_area'])
    return out

p0, p1 = matrix('base_probability'), matrix('complementary_probability')
limit = max(p0.max(), p1.max())
norm = LogNorm(vmin=limit*1e-4, vmax=limit)
density_cmap = DENSITY_CMAP.copy()
density_cmap.set_bad(density_cmap(0))
obs = read('toy_fusion_observations.csv')
truth = (float(obs[0]['truth_mass1']), float(obs[0]['truth_mass2']))
fig = plt.figure(figsize=(6.9, 2.45))
def area(x, y, w, h):
    return fig.add_axes([x/6.9, y/2.45, w/6.9, h/2.45])
axes = np.array([area(x, .5, 1.33, 1.25) for x in [.43, 1.90, 3.82, 5.36]])
def horizontal_scale(mesh, x, width, label, ticks=None):
    cax = area(x, 2.10, width, .055)
    bar = fig.colorbar(mesh, cax=cax, orientation='horizontal', label=label, ticks=ticks)
    cax.xaxis.set_label_position('top')
    cax.tick_params(pad=1, length=2)
    return bar
handles = [Line2D([], [], color=COLORS[1], lw=1., label='50%'),
           Line2D([], [], color=INK, lw=1., ls='--', label='90%'),
           Line2D([], [], color=COLORS[1], marker='+', ls='', ms=7, label='Truth')]
for ax, z, letter in zip(axes[:2], [p0, p1], 'ab'):
    mesh = ax.pcolormesh(edges_from_centers(m1), edges_from_centers(m2),
                         z.T, norm=norm, cmap=density_cmap, shading='flat')
    descending = np.sort(z.ravel())[::-1]
    cumulative = np.cumsum(descending*cell_area)
    cumulative /= cumulative[-1]
    threshold90, threshold50 = [descending[np.searchsorted(cumulative, q)]
                                for q in [.9, .5]]
    contours = ax.contour(m1, m2, z.T, levels=[threshold90, threshold50],
                          colors=[INK, COLORS[1]], linewidths=[.85, 1.],
                          linestyles=['--', '-'])
    contours.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white', alpha=.7)])
    ax.plot(*truth, marker='+', color=COLORS[1], markersize=8, markeredgewidth=1.5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    ax.set(xlabel=r'$m_1$ (a.u.)', ylabel=r'$m_2$ (a.u.)' if letter == 'a' else '',
           xlim=(0, 1.2), ylim=(0, .6), xticks=[0, .4, .8, 1.2],
           yticks=[0, .2, .4, .6])
    ax.set_xticklabels(['0', '0.4', '0.8', '1.2'])
    ax.tick_params(axis='y', labelleft=letter == 'a')
    panel(ax, letter)
horizontal_scale(mesh, .43, 2.80, r'Posterior density (a.u.$^{-2}$)', [1, 10, 100])
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(1.83/6.9, 1.78/2.45),
           ncol=3, frameon=False, borderaxespad=0., handlelength=1.1,
           handletextpad=.35, columnspacing=.7)

base, response = fusion_summaries()
fr = sorted({r['retained_fraction'] for r in response})
timing = sorted({r['recovery_time_relative_error'] for r in response})
gain = np.zeros((len(fr), len(timing)))
reduction = np.zeros_like(gain)
for r in response:
    i = fr.index(r['retained_fraction'])
    j = timing.index(r['recovery_time_relative_error'])
    gain[i, j] = 100*(1-r['crps_mean']/base['crps_mean'])
    reduction[i, j] = 100*(1-r['width90_au']/base['width90_au'])
for ax, z, letter, label in zip(axes[2:], [gain, reduction], 'cd',
                               ['CRPS gain (%)', 'Width gain (%)']):
    im = ax.imshow(z, origin='lower', aspect='auto', cmap=GAIN_CMAP, vmin=0, vmax=45)
    ax.set(xticks=range(5), xticklabels=[f'{100*t:.0f}' for t in timing],
           yticks=range(5), yticklabels=[str(int(round(f*10))) for f in fr],
           xlabel='Timing error (%)', ylabel='Added observations' if letter == 'c' else '')
    ax.tick_params(axis='y', labelleft=letter == 'c')
    ax.tick_params(length=0)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{z[i,j]:.1f}', ha='center', va='center',
                    color=cell_text_color(im.cmap, im.norm, z[i,j]))
    ax.add_patch(Rectangle((1.5, -.5), 1, 5, fill=False, ec=COLORS[1], lw=1.2))
    panel(ax, letter)
    horizontal_scale(im, 3.82 if letter == 'c' else 5.36, 1.33, label, [0, 20, 40])
save(fig, 'Fig6_Evidence_Fusion')
