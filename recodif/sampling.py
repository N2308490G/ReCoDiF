"""Scheduled reverse evolution, conditional refinement, and spectral summaries."""
import numpy as np
import torch
from .cir import positive_reverse_step
from .partitions import hierarchy
from .refinement import split_mass


def prepare_schedule(grid, schedule, terminal):
    """Resolve and validate the trajectory-independent schedule before sampling."""
    if schedule.get('schedule', 'prescribed') != 'prescribed':
        raise ValueError('Only a prescribed, trajectory-independent schedule is implemented.')
    if schedule.get('split_order', 'lexicographic parents; T1 then both T2 children') != 'lexicographic parents; T1 then both T2 children':
        raise ValueError('Required split order: lexicographic parents; T1 then both T2 children.')
    for shape in schedule['grids']:
        if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
            raise ValueError('Partition dimensions must be positive integers.')
    start, stages = hierarchy(grid, schedule['grids'])
    times = list(map(float, schedule['times']))
    step = float(schedule['reverse_step_size'])
    if not np.isfinite([terminal, step]+times).all() or len(times) != len(stages) or step <= 0 or not all(a > b for a, b in zip([terminal]+times, times+[0.])):
        raise ValueError('Require finite T > split times > 0, one time per stage, and a positive reverse step.')
    return start, stages, times, step


@torch.no_grad()
def posterior_draws(model, evidence, schedule, count, seed=0, terminal_draws=None, sample_microbatch=32):
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('Draw count must be positive.')
    if not isinstance(sample_microbatch, (int, np.integer)) or sample_microbatch < 1:
        raise ValueError('Sample microbatch must be a positive integer.')
    terminal = float(model.config['terminal_time'])
    start, stages, times, step = prepare_schedule(model.grid, schedule, terminal)
    rng = np.random.default_rng(seed)
    reference = model.reference_mass.detach().cpu().numpy()
    a = start.aggregate(reference)
    mode = model.training
    model.eval()
    try:
        encoded = model.encode_evidence(evidence)
        def residual(mass, partition, time):
            return np.concatenate([model.residual(mass[i:i+sample_microbatch], partition, time, encoded)
                                   .double().cpu().numpy() for i in range(0, len(mass), sample_microbatch)])
        if terminal_draws is None:
            state = rng.gamma(2*model.kappa*a/model.sigma**2,
                              model.sigma**2/(2*model.kappa), size=(count, start.size))
        else:
            state = np.asarray(terminal_draws, float).copy()
            if state.shape != (count, start.size) or np.any(state < 0) or not np.all(np.isfinite(state)):
                raise ValueError('Terminal draws must match the coarsest partition and draw count.')
        current, time = start, terminal
        options = dict(quadrature_tolerance=float(schedule.get('quadrature_tolerance', 1e-10)),
                       potential_tolerance=float(schedule.get('potential_tolerance', 5e-5)),
                       max_nodes=int(schedule.get('max_quadrature_nodes', 513)))
        for segment, target in enumerate(times+[0.]):
            # Choose equal steps within each segment, landing exactly on a split or zero.
            steps = max(1, int(np.ceil((time-target)/step)))
            dt = (time-target)/steps
            beginning = time
            for index in range(steps):
                time = beginning-index*dt
                state = positive_reverse_step(state, current.aggregate(reference),
                                              residual(state, current, time), dt, model.kappa, model.sigma, rng)
            time = target
            if segment < len(stages):
                for before, after, parent in stages[segment]:
                    if before.cells != current.cells:
                        raise RuntimeError('Partition schedule and current state disagree.')
                    state = np.stack([split_mass(draw, before, after, parent, time, reference,
                                      model.kappa, model.sigma, residual, rng, **options) for draw in state])
                    current = after
        return current.to_reference(state)
    finally:
        model.train(mode)


def summarize(draws, grid):
    x = np.asarray(draws, float)
    if x.ndim != 2 or x.shape[1] != grid.size or len(x) == 0 or np.any(x < 0) or not np.all(np.isfinite(x)):
        raise ValueError('Expected finite nonnegative reference-grid draws.')
    marginal = x.reshape((-1,)+grid.shape).sum(axis=1)/np.diff(grid.t2)
    return {'mean_spectrum': (x.mean(axis=0)/grid.areas).reshape(grid.shape),
            't2_marginal_mean': marginal.mean(axis=0),
            't2_marginal_lower90': np.quantile(marginal, 0.05, axis=0),
            't2_marginal_upper90': np.quantile(marginal, 0.95, axis=0)}
