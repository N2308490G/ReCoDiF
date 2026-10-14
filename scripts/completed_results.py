"""Read the current experiment CSVs for manuscript tables and figures."""
from functools import lru_cache
import csv
import numpy as np
from experiment_data import (ROOT, FIGDIR, TABLEDIR, MACRODIR, DATA_ROOT, PAPER, plt, panel, legend_above, save,
                             INK, DEPTH_COLORS, DENSITY_CMAP, UNCERTAINTY_CMAP,
                             SIGNED_CMAP, ERROR_CMAP, METHOD_COLORS, cell_text_color)

DATA = DATA_ROOT / 'field'
METHODS = ['ReDif', 'DDRM', 'DPS', 'CSGM', 'DAPS']
COLORS = METHOD_COLORS
# Keep CSV identifiers stable while using the current manuscript name in outputs.
def method_label(method):
    return {'ReDif': 'ReCoDiF', 'bNMR_McMC': 'bNMR McMC'}.get(method, method)

VARIANTS = {
    'full_shared_stochastic': 'Full model',
    'no_aggregation_shared_stochastic': 'Without aggregation',
    'aggregation_deterministic_split': 'Deterministic split',
    'no_aggregation_deterministic_split': 'Neither component',
    'independent_conditional_split_head': 'Independent split head',
    'fixed_beta_split': 'Fixed beta split',
    'finest_only_same_field': 'Finest grid only',
}

@lru_cache(maxsize=None)
def rows(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def number(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan

def values(data, key):
    return np.asarray([number(r.get(key)) for r in data])

def primary(method, level='macro', probability=.9):
    return [r for r in rows('performance_by_unit.csv')
            if r['domain'] == 'field' and r['level'] == level
            and r['case_id'] == 'primary' and r['method'] == method
            and (number(r['interval_level']) == probability or method == 'DLEMLR')]

def mean_sd(data, key):
    x = values(data, key)
    x = x[np.isfinite(x)]
    return (x.mean(), x.std(ddof=1)) if len(x) > 1 else (np.nan, np.nan)

def variants():
    data = rows('ablation_results.csv')
    return {k: [r for r in data if r['variant_id'] == k] for k in VARIANTS}

@lru_cache(maxsize=None)
def b10_spectra(seed='1001'):
    """Native cell densities and stored marginal quantiles for the two depths."""
    data = [r for r in rows('method_outputs.csv')
            if r['case_id'] == 'primary' and r['train_seed'] == str(seed)
            and r['record_id'] in ('B10_6-4', 'B10_7-7')]
    output = {}
    for record in ('B10_6-4', 'B10_7-7'):
        for method in METHODS + ['DLEMLR']:
            cells = [r for r in data if r['record_id'] == record and r['method'] == method
                     and r['output_type'] == 'mean_cell_mass']
            left1 = np.sort(np.unique(values(cells, 'log10_t1_left_s')))
            left2 = np.sort(np.unique(values(cells, 'log10_t2_left_s')))
            if len(cells) != len(left1)*len(left2) or len(cells) != 256:
                raise ValueError(f'Incomplete native spectrum: {record}, {method}')
            density = np.full((len(left1), len(left2)), np.nan)
            for cell in cells:
                i = np.searchsorted(left1, number(cell['log10_t1_left_s']))
                j = np.searchsorted(left2, number(cell['log10_t2_left_s']))
                density[i, j] = number(cell['bin_mass'])/number(cell['log_cell_area'])
            marginal = sorted([r for r in data if r['record_id'] == record
                               and r['method'] == method
                               and r['output_type'] == 'mean_T2_marginal_mass'],
                              key=lambda r: number(r['log10_t2_left_s']))
            edges1 = np.r_[left1, max(values(cells, 'log10_t1_right_s'))]
            edges2 = np.r_[left2, max(values(cells, 'log10_t2_right_s'))]
            width = values(marginal, 'log_cell_width')
            marginal_density = values(marginal, 'bin_mass')/width
            if not np.isfinite(density).all() or len(marginal) != len(left2):
                raise ValueError(f'Incomplete spectral summary: {record}, {method}')
            # Integrate over log T1; unequal cell widths must remain explicit.
            if not np.allclose((density*np.diff(edges1)[:, None]).sum(axis=0),
                               marginal_density, rtol=1e-10, atol=1e-10):
                raise ValueError(f'Marginal does not integrate the map: {record}, {method}')
            output[record, method] = dict(
                edges1=edges1, edges2=edges2, density=density,
                marginal=marginal_density,
                q05=values(marginal, 'q05_mass')/width,
                q95=values(marginal, 'q95_mass')/width)
    return output

def tex_mean_sd(data, key, digits=4, factor=1):
    mean, sd = mean_sd(data, key)
    if not np.isfinite(mean):
        return '--'
    return rf'${mean*factor:.{digits}f}\pm{sd*factor:.{digits}f}$'

def table(name, caption, label, headings, body, widths=None):
    n = len(headings)
    alignment = widths or ('l' + 'c'*(n-1))
    tex = [r'\begin{table*}[t]', r'\centering',
           rf'\caption{{{caption}}}\label{{{label}}}', r'\small',
           r'\setlength{\tabcolsep}{5pt}',
           rf'\begin{{tabular}}{{@{{}}{alignment}@{{}}}}', r'\toprule',
           ' & '.join(headings) + r'\\', r'\midrule']
    tex += [' & '.join(line) + r'\\' for line in body]
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    (TABLEDIR / name).write_text('\n'.join(tex), encoding='utf-8')

