"""Shared readers and print styling for the manuscript's CSV-based figures."""
from pathlib import Path
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import QuadMesh
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('RECODIF_DATA_DIR', ROOT / 'data')).resolve()
OUTPUT_ROOT = Path(os.environ.get('RECODIF_OUTPUT_DIR', ROOT / 'outputs')).resolve()
FIGDIR = OUTPUT_ROOT / 'figures'
TABLEDIR = OUTPUT_ROOT / 'tables'
MACRODIR = OUTPUT_ROOT / 'macros'
PAPER = OUTPUT_ROOT
FONT_FAMILY = os.environ.get('RECODIF_FONT', 'Times New Roman')
for directory in (FIGDIR, TABLEDIR, MACRODIR):
    directory.mkdir(parents=True, exist_ok=True)
LABELS = {'candidate_mock': 'C', 'comparator_a_mock': 'A', 'comparator_b_mock': 'B'}
METHODS = list(LABELS)
BLUE, ORANGE, TEAL, VIOLET, GOLD = '#3868A6', '#D97743', '#279A8B', '#8870AD', '#A88B35'
COLORS = [BLUE, ORANGE, TEAL, VIOLET]
METHOD_COLORS = [BLUE, ORANGE, TEAL, VIOLET, GOLD]
INK = '#242C35'
DEPTH_COLORS = [BLUE, ORANGE]
REGIONAL_COLORS = [ORANGE, BLUE, TEAL, VIOLET]
# Blue encodes density, warm coral encodes uncertainty/error, and teal encodes gains.
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    'density_blue', ['#FAFCFF', '#DFEAF7', '#B5CFEA', '#7BA7D1', '#487BB2', '#244C80'])
UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list(
    'uncertainty_coral', ['#FFFCF6', '#FFE8CD', '#FAC694', '#EF9A6D', '#D76955', '#A83E47'])
ERROR_CMAP = UNCERTAINTY_CMAP
GAIN_CMAP = LinearSegmentedColormap.from_list(
    'gain_teal', ['#F8FCFA', '#D7EEE6', '#A1D4C4', '#65B49F', '#348D78', '#16634F'])
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    'signed_blue_coral', ['#355F9D', '#8CAFD7', '#FCFBF8', '#EBA58B', '#BC513B'])
REGION_GROUPS = ['coarse_path_mock', 'fine_aggregated_mock', 'reference_ensemble_a', 'reference_ensemble_b']
REGION_LABELS = [r'$F_c$', r'$F_f$', r'$R_1$', r'$R_2$']
FIGURE_FONT_SIZE = 8.8
font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY),
                      fallback_to_default=False)
plt.rcParams.update({
    'font.family': FONT_FAMILY, 'font.serif': [FONT_FAMILY],
    'font.weight': 'normal', 'axes.labelweight': 'normal',
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#6B737E',
    'xtick.color': INK, 'ytick.color': INK,
    'axes.titleweight': 'normal', 'axes.grid': False,
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT_FAMILY,
    'mathtext.it': FONT_FAMILY + ':italic', 'mathtext.bf': FONT_FAMILY,
    'mathtext.bfit': FONT_FAMILY + ':italic', 'mathtext.cal': FONT_FAMILY + ':italic',
    'mathtext.sf': FONT_FAMILY, 'mathtext.tt': FONT_FAMILY,
    'mathtext.fallback': None,
    'font.size': FIGURE_FONT_SIZE, 'axes.labelsize': FIGURE_FONT_SIZE,
    'xtick.labelsize': FIGURE_FONT_SIZE, 'ytick.labelsize': FIGURE_FONT_SIZE,
    'legend.fontsize': FIGURE_FONT_SIZE,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'lines.linewidth': 1.25, 'pdf.fonttype': 42,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.labelpad': 2.0, 'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
    'figure.constrained_layout.w_pad': .035,
    'figure.constrained_layout.h_pad': .035,
    'figure.constrained_layout.hspace': .025,
    'savefig.dpi': 600, 'savefig.pad_inches': .025,
})

def read(name, folder='realcsv'):
    folder = {'realcsv': 'controlled', 'real_data': 'observed'}.get(folder, folder)
    with (DATA_ROOT / folder / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def supplemented(name):
    with (DATA_ROOT / 'settings' / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def values(rows, key):
    return np.array([float(r[key]) for r in rows])

def edges_from_centers(x, logarithmic=False):
    x = np.asarray(x, float)
    if logarithmic:
        return np.exp(edges_from_centers(np.log(x)))
    middle = (x[:-1] + x[1:]) / 2
    return np.r_[x[0] - (middle[0]-x[0]), middle, x[-1] + (x[-1]-middle[-1])]

def panel(ax, letter):
    label = ax.get_xlabel()
    ax.set_xlabel(f'{label}\n({letter})' if label else f'({letter})',
                  linespacing=1.20, fontweight='normal')


def legend_above(ax, ncol=1, **kwargs):
    options = dict(loc='lower center', bbox_to_anchor=(.5, 1.015),
                   ncol=ncol, frameon=False, borderaxespad=0.,
                   handlelength=1.1, handletextpad=.35, columnspacing=.6,
                   labelspacing=.2, borderpad=.05)
    options.update(kwargs)
    return ax.legend(**options)

def cell_text_color(cmap, norm, value):
    def luminance(color):
        rgb = np.asarray(to_rgb(color))
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        return linear @ np.array([.2126, .7152, .0722])
    background = luminance(cmap(norm(value)))
    light_contrast = 1.05/(background+.05)
    dark_contrast = (background+.05)/(luminance(INK)+.05)
    return 'white' if light_contrast > dark_contrast else INK


def save(fig, stem, directory=FIGDIR, svg=True, font_size=FIGURE_FONT_SIZE):
    fig.align_xlabels()
    for ax in fig.axes:
        ax.grid(False, which='both', axis='both')
        ax.set_title('')
        # Prevent PDF cell-boundary seams; keep text and axes as vector objects.
        for mesh in ax.collections:
            if isinstance(mesh, QuadMesh):
                mesh.set_rasterized(True)
                mesh.set_antialiased(False)
    for obj in fig.findobj(match=Text):
        obj.set_fontfamily(FONT_FAMILY)
        obj.set_fontweight('normal')
        obj.set_fontsize(font_size)
    fig.canvas.draw()
    for obj in fig.findobj(match=Text):
        obj.set_fontfamily(FONT_FAMILY)
        obj.set_fontweight('normal')
        obj.set_fontsize(font_size)
    fig.savefig(directory / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{stem}.png', dpi=220, bbox_inches='tight')
    if svg:
        fig.savefig(directory / f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Generated {stem}.pdf')

def fusion_summaries():
    observations = [r for r in read('toy_fusion_observations.csv')
                    if r['role'] == 'holdout' and r['is_duplicate'] == '0']
    wait, echo, target = [values(observations, key) for key in
                          ['waiting_time_s', 'echo_time_s', 'observed_signal_au']]
    kernel = (1-2*np.exp(-wait[:, None]/np.array([.05, .5]))) * np.exp(-echo[:, None]/np.array([.01, .2]))
    draws = sorted([r for r in read('toy_fusion_draws.csv') if r['scenario_id'] == 'base'],
                   key=lambda r: int(r['draw_id']))
    mass = np.column_stack([values(draws, 'mass1'), values(draws, 'mass2')])
    sigma = float(observations[0]['noise_sigma_au'])
    noise = np.array([np.random.default_rng(int(r['measurement_noise_seed'])).normal(0, sigma, len(target)) for r in draws])
    pred = mass @ kernel.T + noise
    n = len(pred)
    crps = (np.abs(pred-target).mean(axis=0)
            - ((2*np.arange(n)-n+1)[:, None]*np.sort(pred, axis=0)).sum(axis=0)/n**2)
    lo, hi = np.quantile(pred, [.05, .95], axis=0)
    base = {'scenario_id':'base', 'added_context_count':0, 'context_count':4,
            'retained_fraction':0., 'recovery_time_relative_error':0.,
            'crps_mean':float(crps.mean()), 'coverage90':float(((target>=lo)&(target<=hi)).mean()),
            'width90_au':float((hi-lo).mean()), 'draw_count':n, 'holdout_count':len(target)}
    response = [{k:(r[k] if k in ('scenario_id','source_type') else float(r[k])) for k in r}
                for r in read('toy_fusion_response.csv')]
    return base, response

def split_statistics():
    rows = read('toy_split_cells.csv')
    output = []
    for mass in sorted({float(r['parent_mass']) for r in rows}):
        subset = sorted([r for r in rows if float(r['parent_mass']) == mass], key=lambda r: float(r['ratio_left']))
        p = values(subset, 'reference_probability')
        q = values(subset, 'sample_count') / values(subset, 'sample_size')
        output.append({'mass':mass, 'cdf':float(np.abs(np.cumsum(q-p)).max()),
                       'tv':float(.5*np.abs(q-p).sum()), 'rows':subset})
    return output
