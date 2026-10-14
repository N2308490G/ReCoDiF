"""Model, reference posterior and demonstration command-line entry points."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_case, load_manifest, DATASET_URL, DATASET_DOWNLOAD_URL
from .evidence import Evidence, nmr_kernel, predictive_distribution
from .partitions import Grid
from .reference import posterior_references, reference_diagnostics, select_reference_draws
from .sampling import posterior_draws, summarize
from .training import train, model_from_state

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def infer(checkpoint, case_path, output, count, seed, device, schedule_override=None, sample_microbatch=32):
    output = Path(output)
    if output.suffix != '.npz' or output.exists():
        raise ValueError('Choose a new inference output path ending in .npz.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    count = int(saved['training_config']['test_draw_count']) if count is None else count
    model = model_from_state(saved, device)
    case = load_case(case_path)
    if not np.array_equal(case.grid.t1, model.grid.t1) or not np.array_equal(case.grid.t2, model.grid.t2):
        raise ValueError('Inference must use the checkpoint reference grid.')
    schedule = saved['schedule'] if schedule_override is None else schedule_override
    draws = posterior_draws(model, case.evidence, schedule, count, seed, sample_microbatch=sample_microbatch)
    arrays = dict(draws=draws, log_t1_edges=case.grid.t1, log_t2_edges=case.grid.t2, **summarize(draws, case.grid))
    if case.heldout is not None:
        y, kernel, hh, ho = case.heldout
        means, covariance = predictive_distribution(case.evidence, draws, kernel, hh, ho)
        noise = np.random.default_rng(seed+1).multivariate_normal(np.zeros(len(y)), covariance, size=count)
        arrays['predictive_draws'] = means+noise
        arrays['predictive_mean'] = means.mean(0)
        arrays['heldout_y'] = y
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f'Saved {count} posterior draws and spectral summaries: {output}', flush=True)
    return arrays


def run_reference(args):
    if args.out.suffix != '.npz' or args.out.exists():
        raise ValueError('Choose a new reference output path ending in .npz.')
    case = load_case(args.case)
    protocol = read_json(args.protocol)
    if args.draws < args.chains or args.draws > args.chains*args.draws_per_chain:
        raise ValueError('The total training budget must span all chains and not exceed saved chain draws.')
    samples = posterior_references(case.evidence, case.grid, protocol['prior'], args.draws_per_chain,
                                    args.burnin, args.chains, args.thinning, args.seed)
    report = reference_diagnostics(samples, case.grid, protocol['regional_log10_edges_s'])
    selected, indices = select_reference_draws(samples, args.draws, args.seed)
    with np.load(args.case, allow_pickle=False) as original:
        arrays = {key: original[key] for key in original.files}
    _, keep = Evidence.retained(arrays['y'], arrays['K'], arrays['covariance'],
                                arrays.get('retained_mask'), arrays.get('observation_ids'))
    arrays.update(reference_draws=selected, reference_chain_draws=samples, reference_selected_indices=indices,
                  reference_retained_indices=keep,
                  reference_diagnostics_json=np.array(json.dumps(report)),
                  reference_settings_json=np.array(json.dumps({'prior': protocol['prior'], 'sampler': 'elliptical_slice',
                     'chains': args.chains, 'draws_per_chain': args.draws_per_chain,
                     'training_draw_count': args.draws, 'selection': 'chain_balanced_temporal_strata', 'burnin': args.burnin,
                     'thinning': args.thinning, 'seed': args.seed})))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Saved {args.draws} training references and all {args.chains} x {args.draws_per_chain} chain draws: {args.out}', flush=True)
    # Retain the sampled chains even if convergence criteria are unmet.
    return 0 if report['criteria_met'] else 2


def make_demo(out, seed=123):
    """Small newly sampled controlled example; never reads/writes paper results."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid(np.array([-3., -1., 1.]), np.array([-3., -1., 1.]))
    prior = read_json(ROOT/'configs/protocol.json')['prior']
    rng = np.random.default_rng(seed)
    w, t = np.meshgrid([.005, .05, .5, 2.], [.002, .02, .2], indexing='ij')
    kernel = nmr_kernel(grid, w.ravel(), t.ravel())
    kept = np.ones(len(kernel), dtype=bool)
    kept[[2, 7, 10]] = False
    covariance = np.eye(len(kernel))*.08**2
    entries = []
    for index, role in enumerate(('train', 'train', 'validation')):
        truth = rng.lognormal(-1.8, .4, grid.size)
        y = kernel @ truth+rng.multivariate_normal(np.zeros(len(kernel)), covariance)
        evidence = Evidence(y[kept], kernel[kept], covariance[np.ix_(kept, kept)])
        arrays = dict(y=evidence.y, K=evidence.kernel, covariance=evidence.covariance,
                      log_t1_edges=grid.t1, log_t2_edges=grid.t2,
                      y_holdout=y[~kept], K_holdout=kernel[~kept], covariance_hh=covariance[np.ix_(~kept, ~kept)],
                      covariance_ho=covariance[np.ix_(~kept, kept)],
                      heldout_train_ids=np.repeat(np.arange(4), 3)[~kept])
        if role == 'train':
            arrays['reference_draws'] = posterior_references(evidence, grid, prior,
                                      draws_per_chain=128, burnin=300, chains=4, seed=seed+index)
            arrays['reference_retained_indices'] = np.arange(kept.sum())
        name = f'case_{index}.npz'
        np.savez_compressed(out/name, **arrays)
        entries.append(dict(record_id=f'demo_{index}', site_id=f'demo_site_{index}', well_id=f'demo_well_{index}',
                            mask_id='0', role=role, scale=1., path=name))
    write_json(out/'manifest.json', {'description': 'New small controlled software example, separate from manuscript experiments.',
                                    'cases': entries})
    model = read_json(ROOT/'configs/model.json')
    model.update(width=32, heads=4, blocks=1, terminal_time=1.5, minimum_training_time=.03)
    training = read_json(ROOT/'configs/training.json')
    training.update(max_updates=200, lr_warmup_updates=20, denoising_only_warmup_updates=40,
                    learning_rate=.001, minimum_learning_rate=.00005, validation_interval_updates=100,
                    validation_draw_count=8, effective_batch_size=32)
    schedule = dict(grids=[[1, 1], [2, 2]], times=[.4], reverse_step_size=.1,
                    quadrature_tolerance=1e-8, potential_tolerance=1e-4, max_quadrature_nodes=513)
    write_json(out/'model.json', model)
    write_json(out/'training.json', training)
    write_json(out/'schedule.json', schedule)
    return model, training, schedule


def run_demo(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Choose an empty demo output directory.')
    model, settings, schedule = make_demo(args.out, args.seed)
    if args.steps is not None:
        if args.steps < 2:
            raise ValueError('A demo requires at least two training updates.')
        settings.update(max_updates=args.steps, validation_interval_updates=args.steps,
                        denoising_only_warmup_updates=min(40, args.steps//2), lr_warmup_updates=min(20, args.steps//4))
        write_json(args.out/'training.json', settings)
    cases = load_manifest(args.out/'manifest.json')
    trainer = train(cases, model, settings, schedule, args.out/'run', args.seed, args.device,
                    sample_microbatch=args.sample_microbatch)
    arrays = infer(args.out/'run/best.pt', args.out/'case_2.npz', args.out/'posterior.npz',
                   16, args.seed+99, args.device, sample_microbatch=args.sample_microbatch)
    summary = {'example': 'New controlled end-to-end software demonstration; not a manuscript benchmark.',
               'updates': trainer.update, 'best_update': trainer.best_update,
               'validation_normalized_rmse': trainer.best_score,
               'posterior_shape': list(arrays['draws'].shape), 'all_finite': bool(np.isfinite(arrays['draws']).all()),
               'minimum_mass': float(arrays['draws'].min())}
    write_json(args.out/'demo_result.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = read_json(ROOT/'configs/runtime.json')
    defaults = read_json(ROOT/'configs/training.json')
    commands = parser.add_subparsers(dest='command', required=True)
    download_help = f'Raw dataset: {DATASET_URL}\nZIP: {DATASET_DOWNLOAD_URL}\nPrepare NPZ cases as described in docs/MODEL_IMPLEMENTATION.md.'
    training = commands.add_parser('train', help='Train and select the online model from a case manifest.',
                                   epilog=download_help, formatter_class=argparse.RawDescriptionHelpFormatter)
    training.add_argument('--manifest', type=Path, required=True, help='Prepared fold JSON; raw-data download links are below.')
    training.add_argument('--model-config', type=Path, default=ROOT/'configs/model.json')
    training.add_argument('--training-config', type=Path, default=ROOT/'configs/training.json')
    training.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol.json')
    training.add_argument('--schedule', type=Path, help='Override only the prescribed refinement schedule.')
    training.add_argument('--out', type=Path, required=True)
    training.add_argument('--resume', type=Path)
    training.add_argument('--stop-after', type=int, help='Stop at this absolute update without changing the LR schedule.')
    inference = commands.add_parser('infer', help='Draw spectra using the online weights stored in a checkpoint.',
                                    epilog=download_help, formatter_class=argparse.RawDescriptionHelpFormatter)
    inference.add_argument('--checkpoint', type=Path, required=True)
    inference.add_argument('--case', type=Path, required=True)
    inference.add_argument('--out', type=Path, required=True)
    inference.add_argument('--draws', type=int, help='Default: checkpoint test_draw_count (paper: 512).')
    inference.add_argument('--schedule', type=Path)
    reference = commands.add_parser('reference', help='Sample the fixed-likelihood lognormal reference posterior.',
                                    epilog=download_help, formatter_class=argparse.RawDescriptionHelpFormatter)
    reference.add_argument('--case', type=Path, required=True)
    reference.add_argument('--out', type=Path, required=True)
    reference.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol.json')
    reference.add_argument('--chains', type=int, default=4)
    reference.add_argument('--draws', type=int, default=defaults['reference_draw_count_per_evidence'],
                           help='Total selected training references per evidence set (paper: 512).')
    reference.add_argument('--draws-per-chain', type=int, default=512,
                           help='Full saved draws per chain for diagnostics, retained separately (default: 512).')
    reference.add_argument('--burnin', type=int, default=1000)
    reference.add_argument('--thinning', type=int, default=1)
    demo = commands.add_parser('demo', help='Run a small fresh reference/train/infer example.')
    demo.add_argument('--out', type=Path, default=ROOT/'outputs/model_demo')
    demo.add_argument('--steps', type=int, help='Override the 200-update demo budget.')
    for command in (training, inference, reference, demo):
        command.add_argument('--seed', type=int, default=1001)
        command.add_argument('--threads', type=int, default=int(runtime['intraop_threads']))
        command.add_argument('--interop-threads', type=int, default=int(runtime['interop_threads']))
        if command is not reference:
            command.add_argument('--device', default='cpu')
            command.add_argument('--sample-microbatch', type=int, default=int(runtime['sample_microbatch']))
    args = parser.parse_args(argv)
    if args.threads < 1 or args.interop_threads < 1:
        parser.error('Thread counts must be positive')
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != args.interop_threads:
        torch.set_num_interop_threads(args.interop_threads)
    if args.command == 'reference':
        return run_reference(args)
    if args.device != 'cpu' and not args.device.startswith('cuda'):
        parser.error('Supported devices are cpu and cuda[:index].')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable.')
    if args.command == 'train':
        cases = load_manifest(args.manifest)
        schedule = read_json(args.schedule) if args.schedule else read_json(args.protocol)['refinement']
        train(cases, read_json(args.model_config), read_json(args.training_config), schedule,
              args.out, args.seed, args.device, args.resume, args.stop_after, args.sample_microbatch)
        return 0
    if args.command == 'infer':
        infer(args.checkpoint, args.case, args.out, args.draws, args.seed, args.device,
              read_json(args.schedule) if args.schedule else None, args.sample_microbatch)
        return 0
    return run_demo(args)
