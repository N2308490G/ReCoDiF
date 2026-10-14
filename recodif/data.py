"""Explicit NPZ evidence/reference inputs; no inference of draws from summaries."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .evidence import Evidence
from .partitions import Grid

# Original borehole/laboratory acquisition archive; local NPZ cases are prepared
# from these raw files following docs/MODEL_IMPLEMENTATION.md.
DATASET_URL = 'https://zenodo.org/records/14947500'
DATASET_DOWNLOAD_URL = ('https://zenodo.org/api/records/14947500/files/'
                        'Mashhadi%20et%20al%20(T1-T2%20Data).zip/content')


@dataclass
class Case:
    record_id: str
    site_id: str
    role: str
    scale: float
    grid: Grid
    evidence: Evidence
    references: object = None
    heldout: object = None
    well_id: object = None
    mask_id: str = '0'
    heldout_train_ids: object = None


def load_case(path, record_id=None, site_id='unspecified', role='inference', scale=1.0,
              well_id=None, mask_id='0'):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Prepared case not found: {path}. Download raw data from {DATASET_URL}; '
                                'see docs/MODEL_IMPLEMENTATION.md for the NPZ input format.')
    with np.load(path, allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    for key in ('y', 'K', 'covariance', 'log_t1_edges', 'log_t2_edges'):
        if key not in arrays:
            raise ValueError(f'{path.name}: missing {key}.')
    grid = Grid(arrays['log_t1_edges'], arrays['log_t2_edges'])
    evidence, keep = Evidence.retained(arrays['y'], arrays['K'], arrays['covariance'],
                                       arrays.get('retained_mask'), arrays.get('observation_ids'))
    if evidence.kernel.shape[1] != grid.size:
        raise ValueError('Kernel columns must match reference-grid cells.')
    draws = arrays.get('reference_draws')
    if draws is not None:
        if draws.ndim not in (2, 3) or draws.shape[-1] != grid.size or np.any(draws < 0) or not np.all(np.isfinite(draws)):
            raise ValueError('reference_draws must be [draw,cells] or [chain,draw,cells], finite and nonnegative.')
        draws = np.asarray(draws, dtype=float).reshape(-1, grid.size)
        if not len(draws):
            raise ValueError('Empty reference draws.')
        if 'reference_retained_indices' in arrays and not np.array_equal(arrays['reference_retained_indices'], keep):
            raise ValueError('The retained mask differs from the reference sampler conditioning set.')
    heldout_keys = ('y_holdout', 'K_holdout', 'covariance_hh')
    present = [k in arrays for k in heldout_keys]
    heldout = None
    train_ids = arrays.get('heldout_train_ids')
    if any(present):
        if not all(present):
            raise ValueError('Provide all held-out y, K and covariance_hh arrays.')
        yh, kh, hh = (np.asarray(arrays[k], float) for k in heldout_keys)
        ho = np.asarray(arrays.get('covariance_ho', np.zeros((len(yh), len(arrays['y'])))), float)
        if yh.ndim != 1 or not len(yh) or kh.shape != (len(yh), grid.size) or hh.shape != (len(yh), len(yh)) or ho.shape != (len(yh), len(arrays['y'])):
            raise ValueError('Invalid held-out array dimensions; covariance_ho must index original observation rows.')
        if not all(np.all(np.isfinite(v)) for v in (yh, kh, hh, ho)):
            raise ValueError('Held-out arrays must be finite.')
        heldout = (yh, kh, hh, ho[:, keep])
        if train_ids is not None and (train_ids.shape != yh.shape or train_ids.dtype.kind not in 'USiu'
                                      or any(not str(v) for v in train_ids)):
            raise ValueError('heldout_train_ids must identify the recovery train of each held-out echo.')
    elif train_ids is not None:
        raise ValueError('heldout_train_ids requires held-out observations.')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Metric normalization scale must be positive.')
    return Case(record_id or path.stem, str(site_id), role, float(scale), grid, evidence, draws, heldout,
                None if well_id is None else str(well_id), str(mask_id), train_ids)


def load_manifest(path, roles=('train', 'validation')):
    path = Path(path).resolve()
    entries = json.loads(path.read_text(encoding='utf-8'))['cases']
    records, site_roles, cases = set(), {}, []
    for entry in entries:
        identifier, site, role = entry['record_id'], entry['site_id'], entry['role']
        if role not in ('train', 'validation', 'test', 'recalibration'):
            raise ValueError('Unknown fold role: '+role)
        key = (site, entry.get('well_id'), identifier, str(entry.get('mask_id', '0')))
        if key in records:
            raise ValueError('Repeated site/well/record/mask in manifest: '+identifier)
        records.add(key)
        if site in site_roles and site_roles[site] != role:
            raise ValueError('A site cannot span training, validation, test or recalibration roles.')
        site_roles[site] = role
    # Validate metadata before opening any observation files.
    for entry in entries:
        if entry['role'] in roles:
            cases.append(load_case(path.parent/entry['path'], entry['record_id'], entry['site_id'],
                         entry['role'], entry.get('scale', 1.), entry.get('well_id'), entry.get('mask_id', '0')))
    if not cases:
        raise ValueError('No cases for the selected roles.')
    first = cases[0].grid
    for case in cases:
        if not np.array_equal(case.grid.t1, first.t1) or not np.array_equal(case.grid.t2, first.t2):
            raise ValueError('A trained model uses one common finest reference grid.')
        if case.role == 'train' and case.references is None:
            raise ValueError(f'{case.record_id}: posterior reference draws are required for training.')
        if case.role == 'validation' and case.heldout is None:
            raise ValueError(f'{case.record_id}: held-out observations are required for model selection.')
        if case.role == 'validation' and (case.heldout_train_ids is None or case.well_id is None):
            raise ValueError(f'{case.record_id}: validation needs heldout_train_ids and well_id '
                             'for train/depth/well/site averaging. Use a specimen ID for laboratory data.')
    return cases
