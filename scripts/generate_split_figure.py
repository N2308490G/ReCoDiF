"""Atlas plus diagnostics: allocation changes with parent amplitude.

Three aligned density maps establish the response, a wide slice panel resolves
the distributions, and all 32 CDF/TV errors quantify sampling agreement.
"""
from experiment_data import *
from matplotlib.colors import TwoSlopeNorm, PowerNorm

stats = split_statistics()
masses = np.array([r['mass'] for r in stats])
me = edges_from_centers(masses)
reference = np.array([values(r['rows'], 'reference_density') for r in stats])
empirical = np.array([values(r['rows'], 'empirical_density') for r in stats])
first = stats[0]['rows']
se = np.r_[values(first, 'ratio_left'), float(first[-1]['ratio_right'])]
vmax = max(reference.max(), empirical.max())
delta = empirical-reference
lim = abs(delta).max()
fig = plt.figure(figsize=(6.9, 3.65))
def area(x,y,w,h):
    return fig.add_axes([x/6.9,y/3.65,w/6.9,h/3.65])
def horizontal_scale(mesh,x,w,label,ticks=None):
    cax=area(x,3.26,w,.055)
    fig.colorbar(mesh,cax=cax,orientation='horizontal',label=label,ticks=ticks)
    cax.xaxis.set_label_position('top')
    cax.tick_params(pad=1,length=2)
top_axes = [area(x,2.06,1.84,1.05) for x in [.43,2.61,4.79]]
for j, (z, letter) in enumerate(zip([reference, empirical, delta], 'abc')):
    ax = top_axes[j]
    kw = ({'norm': TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
           'cmap': SIGNED_CMAP} if letter == 'c'
          else {'norm': PowerNorm(.5, vmin=0, vmax=vmax), 'cmap': DENSITY_CMAP})
    mesh = ax.pcolormesh(se, me, z, shading='flat', **kw)
    ax.set(xlabel=r'Split fraction $s=x_1/M$', ylabel=r'Parent mass $M$ (a.u.)' if j==0 else '',
           xticks=[0, .5, 1], yticks=[.2, .6, 1, 1.4])
    ax.tick_params(axis='y', labelleft=j==0)
    panel(ax, letter)
    if j==1:
        horizontal_scale(mesh,.43,4.02,'Bin-average density',[0,1,5,10,15])
    if j==2:
        horizontal_scale(mesh,4.79,1.84,'Density difference',[-1.5,0,1.5])

ax = area(.43,.48,3.77,.84)
centers = (se[:-1]+se[1:])/2
selected = [min(stats, key=lambda r: abs(r['mass']-target)) for target in [.6, 1., 1.4]]
for r, color in zip(selected, [COLORS[0], COLORS[2], COLORS[1]]):
    density = values(r['rows'], 'reference_density')
    observed = values(r['rows'], 'empirical_density')
    ax.stairs(density, se, color=color, lw=1.25, label=rf'$M={r["mass"]:.3f}$')
    ax.stairs(density, se, color=color, fill=True, alpha=.07, lw=0)
    ax.scatter(centers, observed, s=9, facecolor='white', edgecolor=color,
               linewidth=.65, zorder=3)
ax.set(xlabel=r'Split fraction $s=x_1/M$', ylabel='Bin-average density',
       xlim=(0, 1), ylim=(0, None))
legend_above(ax, ncol=3)
panel(ax, 'd')

ax = area(4.98,.48,1.66,.84)
for key, color, marker, label in [('cdf', COLORS[0], 'o', 'CDF'),
                                 ('tv', COLORS[1], 's', 'TV')]:
    ax.plot(masses, [r[key] for r in stats], color=color, marker=marker,
            ms=2.6, lw=1., label=label)
ax.set(xlabel=r'Parent mass $M$ (a.u.)', ylabel='Error',
       xlim=(.16, 1.44), ylim=(0, .1), xticks=[.2, .6, 1, 1.4],
       yticks=[0, .05, .1])
legend_above(ax, ncol=2)
panel(ax, 'e')
save(fig, 'Fig5_Conditional_Split')
