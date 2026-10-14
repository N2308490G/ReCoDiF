"""Paired CIR denoising, stop-gradient EMA regression, selection and resume."""
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from .cir import corrupt, denoising_target
from .evidence import predictive_distribution
from .metrics import train_mean_rmse, site_macro_mean
from .model import SharedFlux
from .partitions import Grid, aggregation_matrix
from .sampling import posterior_draws, prepare_schedule

SELECTION_METRIC = 'per_train_rmse_mask_depth_well_site_v1'


def learning_rate(update, config):
    warm, total = config['lr_warmup_updates'], config['max_updates']
    peak, floor = config['learning_rate'], config['minimum_learning_rate']
    if warm and update < warm:
        return peak*(update+1)/warm
    fraction = min(1., (update-warm)/max(1, total-warm-1))
    return floor+.5*(peak-floor)*(1+math.cos(math.pi*fraction))


class Trainer:
    def __init__(self, model, config, schedule, seed, sample_microbatch=32):
        self.model, self.config, self.schedule = model, dict(config), deepcopy(schedule)
        if not isinstance(sample_microbatch, int) or sample_microbatch < 1:
            raise ValueError('Sample microbatch must be a positive integer.')
        self.sample_microbatch = sample_microbatch
        if config['optimizer'] != 'AdamW' or config['lr_schedule'] != 'linear_warmup_then_cosine_decay':
            raise ValueError('This implementation supports the documented AdamW/cosine schedule.')
        if model.config['time_weight'] != 'constant':
            raise ValueError('Only the explicit constant positive time weight is implemented.')
        if not 0 < model.config['minimum_training_time'] < model.config['terminal_time']:
            raise ValueError('Require 0 < minimum training time < terminal time.')
        if not 0 <= config['teacher_ema'] < 1 or config['aggregation_lambda'] < 0 or config['effective_batch_size'] < 1:
            raise ValueError('Invalid training configuration.')
        if config['max_updates'] < 1 or config['validation_interval_updates'] < 1 or config['denoising_only_warmup_updates'] < 0:
            raise ValueError('Invalid training/update budget.')
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                           betas=tuple(config['betas']), eps=config['epsilon'], weight_decay=config['weight_decay'])
        self.teacher = None
        self.rng = np.random.default_rng(seed)
        self.seed, self.update = int(seed), 0
        self.best_score, self.best_update = math.inf, None
        self.best_online = None
        start, stages, _, _ = prepare_schedule(model.grid, schedule, model.config['terminal_time'])
        self.pairs = [(after, before, aggregation_matrix(after, before))
                      for stage in stages for before, after, _ in stage]
        if not self.pairs:
            raise ValueError('Cross-resolution training requires at least one refinement stage.')
        # Both mixed binary states and the full adjacent resolution pairs occur.
        previous = start
        for stage in stages:
            last = stage[-1][1]
            self.pairs.append((last, previous, aggregation_matrix(last, previous)))
            previous = last

    def _initialize_teacher(self):
        self.teacher = deepcopy(self.model).eval()
        self.teacher.requires_grad_(False)

    def batch(self, case):
        if case.references is None:
            raise ValueError('Training requires actual posterior reference draws.')
        if len(case.references) != self.config['reference_draw_count_per_evidence']:
            raise ValueError(f'{case.record_id}: expected exactly {self.config["reference_draw_count_per_evidence"]} '
                             'reference draws per evidence set; prepare a fixed balanced subset with the reference command.')
        fine, coarse, matrix = self.pairs[int(self.rng.integers(len(self.pairs)))]
        count = int(self.config['effective_batch_size'])
        indices = self.rng.integers(len(case.references), size=count)
        initial = fine.aggregate(case.references[indices])
        a = self.model.partition_reference(fine)
        time = self.rng.uniform(self.model.config['minimum_training_time'], self.model.config['terminal_time'], (count, 1))
        xt = corrupt(initial, a, time, self.model.kappa, self.model.sigma, self.rng)
        coarse_state, coarse_initial = xt @ matrix.T, initial @ matrix.T
        target_f = denoising_target(xt, initial, a, time, self.model.kappa, self.model.sigma)
        target_c = denoising_target(coarse_state, coarse_initial, self.model.partition_reference(coarse), time,
                                    self.model.kappa, self.model.sigma)
        return fine, coarse, matrix, xt, coarse_state, time[:, 0], target_f, target_c

    def losses(self, evidence, batch, aggregation):
        fine, coarse, matrix, xt, ct, time, target_f, target_c = batch
        model = self.model
        def tensor(x):
            return torch.as_tensor(x, dtype=model.dtype, device=model.device)
        encoded = model.encode_evidence(evidence)
        rf = model(xt, fine, time, encoded)
        rc = model(ct, coarse, time, encoded)
        # Squared L2 norm over cells, expectation over examples; equal resolution weight.
        den = .5*((rf-tensor(target_f)).square().sum(-1).mean() +
                  (rc-tensor(target_c)).square().sum(-1).mean())
        agg = den.new_zeros(())
        if aggregation:
            if self.teacher is None:
                raise RuntimeError('Initialize the EMA teacher before computing aggregation loss.')
            with torch.no_grad():
                encoded_teacher = self.teacher.encode_evidence(evidence)
                target = self.teacher(xt, fine, time, encoded_teacher) @ tensor(matrix).T
            agg = (rc-target).square().sum(-1).mean()
        return den, agg

    def step(self, case):
        self.model.train()
        active = self.update >= self.config['denoising_only_warmup_updates']
        if active and self.teacher is None:
            self._initialize_teacher()
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate(self.update, self.config)
        self.optimizer.zero_grad(set_to_none=True)
        den, agg = self.losses(case.evidence, self.batch(case), active)
        loss = den+self.config['aggregation_lambda']*agg
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_norm'], error_if_nonfinite=True)
        self.optimizer.step()
        if active:
            with torch.no_grad():
                eta = self.config['teacher_ema']
                for teacher, online in zip(self.teacher.parameters(), self.model.parameters()):
                    teacher.mul_(eta).add_(online, alpha=1-eta)
                for teacher, online in zip(self.teacher.buffers(), self.model.buffers()):
                    teacher.copy_(online)
        self.update += 1
        return {'update': self.update, 'loss': float(loss.detach()), 'denoising': float(den.detach()),
                'aggregation': float(agg.detach()), 'gradient_norm': float(norm),
                'learning_rate': self.optimizer.param_groups[0]['lr']}

    def validate(self, cases):
        scores = []
        for i, case in enumerate(cases):
            if case.role != 'validation' or case.heldout is None:
                raise ValueError('Model selection requires held-out observations on validation cases only.')
            if case.heldout_train_ids is None or case.well_id is None:
                raise ValueError('Validation requires heldout_train_ids and well_id for the paper aggregation rule.')
            draws = posterior_draws(self.model, case.evidence, self.schedule,
                                    self.config['validation_draw_count'], seed=self.seed+100000+i,
                                    sample_microbatch=self.sample_microbatch)
            y, kernel, hh, ho = case.heldout
            mean, _ = predictive_distribution(case.evidence, draws, kernel, hh, ho)
            score = train_mean_rmse(mean.mean(0), y, case.heldout_train_ids, case.scale)
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite validation score.')
            scores.append((case.site_id, case.well_id, case.record_id, case.mask_id, score))
        if not scores:
            raise ValueError('At least one validation site is required.')
        return site_macro_mean(scores)

    def select(self, score):
        """Retain online weights; an exact tie leaves the earlier selection intact."""
        if not np.isfinite(score):
            raise ValueError('Validation score must be finite.')
        if score >= self.best_score:
            return False
        self.best_score, self.best_update = float(score), self.update
        self.best_online = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        return True

    def save_best(self, path):
        if self.best_online is None:
            raise ValueError('No validation-selected online weights are available.')
        state = {'format_version': 1, 'inference_only': True, 'online': self.best_online,
                 'model_config': self.model.config, 'training_config': self.config, 'schedule': self.schedule,
                 'grid_t1': self.model.grid.t1.tolist(), 'grid_t2': self.model.grid.t2.tolist(),
                 'update': self.best_update, 'best_update': self.best_update, 'best_score': self.best_score,
                 'seed': self.seed, 'sample_microbatch': self.sample_microbatch,
                 'selection_metric': SELECTION_METRIC}
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(state, temporary)
        os.replace(temporary, path)

    def save(self, path, data_keys=None):
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                                                stderr=subprocess.DEVNULL, text=True).strip()
            modified = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[1],
                                                   stderr=subprocess.DEVNULL, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            revision = 'unavailable'
            modified = None
        state = {'format_version': 1, 'model_config': self.model.config, 'training_config': self.config,
                 'schedule': self.schedule, 'grid_t1': self.model.grid.t1.tolist(), 'grid_t2': self.model.grid.t2.tolist(),
                 'online': self.model.state_dict(), 'teacher': None if self.teacher is None else self.teacher.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'update': self.update, 'seed': self.seed,
                 'best_score': self.best_score, 'best_update': self.best_update,
                 'best_online': self.best_online, 'sample_microbatch': self.sample_microbatch,
                 'selection_metric': SELECTION_METRIC,
                 'numpy_rng_json': json.dumps(self.rng.bit_generator.state), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if self.model.device.type == 'cuda' else [],
                 'data_keys': data_keys, 'code_revision': revision, 'code_modified': modified,
                 'environment': {'torch': str(torch.__version__), 'numpy': str(np.__version__),
                                 'device': str(self.model.device), 'threads': torch.get_num_threads()}}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(state, temporary)
        os.replace(temporary, path)

    @classmethod
    def resume(cls, path, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('inference_only', False):
            raise ValueError('best.pt contains selected inference weights; resume training from latest.pt.')
        if state.get('best_update') is not None and state.get('selection_metric') != SELECTION_METRIC:
            raise ValueError('This checkpoint used the earlier flattened validation RMSE; its selected score '
                             'cannot be compared with the corrected train/depth/well/site metric. '
                             'Existing weights remain usable for inference; start a new training run for corrected selection.')
        model = model_from_state(state, device)
        trainer = cls(model, state['training_config'], state['schedule'], state['seed'], state.get('sample_microbatch', 32))
        trainer.optimizer.load_state_dict(state['optimizer'])
        if state['teacher'] is not None:
            trainer._initialize_teacher()
            trainer.teacher.load_state_dict(state['teacher'])
        trainer.update, trainer.best_score, trainer.best_update = state['update'], state['best_score'], state['best_update']
        trainer.best_online = state.get('best_online')
        if trainer.best_update is not None and trainer.best_online is None:
            # Load a legacy selection without substituting the latest weights.
            legacy_path = Path(path).with_name('best.pt')
            if trainer.update == trainer.best_update:
                selected = state
            elif legacy_path.is_file():
                selected = torch.load(legacy_path, map_location='cpu', weights_only=True)
                if selected['update'] != trainer.best_update or selected['best_score'] != trainer.best_score:
                    raise ValueError('Legacy best.pt does not match the saved validation selection.')
            else:
                raise ValueError('Legacy resume requires its companion best.pt to retain the selected online weights.')
            trainer.best_online = {k: v.cpu().clone() for k, v in selected['online'].items()}
        trainer.rng.bit_generator.state = json.loads(state['numpy_rng_json'])
        torch.set_rng_state(state['torch_rng'])
        if device.startswith('cuda') and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        return trainer, state.get('data_keys')


def model_from_state(state, device='cpu'):
    if state.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format.')
    grid = Grid(state['grid_t1'], state['grid_t2'])
    model = SharedFlux(grid, state['model_config']).to(device)
    model.load_state_dict(state['online'])
    return model


def train(cases, model_config, training_config, schedule, out, seed=1001, device='cpu', resume=None, stop_after=None,
          sample_microbatch=32):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    training = [case for case in cases if case.role == 'train']
    validation = [case for case in cases if case.role == 'validation']
    if not training or not validation:
        raise ValueError('Supply both training and validation cases.')
    for case in training:
        if case.references is None or len(case.references) != training_config['reference_draw_count_per_evidence']:
            raise ValueError(f'{case.record_id}: reference count must equal reference_draw_count_per_evidence.')
    for case in validation:
        if case.heldout is None or case.heldout_train_ids is None or case.well_id is None:
            raise ValueError(f'{case.record_id}: validation needs held-out observations, recovery-train IDs and well_id.')
    keys = [(c.record_id, c.site_id, c.role, c.well_id, c.mask_id) for c in cases]
    if resume:
        trainer, saved_keys = Trainer.resume(resume, device)
        legacy_keys = [(c.record_id, c.site_id, c.role) for c in cases]
        if saved_keys not in (keys, legacy_keys) or trainer.config != training_config or trainer.model.config != model_config or trainer.schedule != schedule or trainer.sample_microbatch != sample_microbatch:
            raise ValueError('Resume requires unchanged data roles/order and saved configurations.')
        if not np.array_equal(trainer.model.grid.t1, training[0].grid.t1) or not np.array_equal(trainer.model.grid.t2, training[0].grid.t2):
            raise ValueError('Resume reference grid changed.')
    else:
        if any((out/name).exists() for name in ('history.csv', 'latest.pt', 'best.pt')):
            raise FileExistsError('Run directory already contains training output; resume or choose a new directory.')
        torch.manual_seed(seed)
        trainer = Trainer(SharedFlux(training[0].grid, model_config).to(device), training_config, schedule, seed, sample_microbatch)
    stop = training_config['max_updates'] if stop_after is None else min(int(stop_after), training_config['max_updates'])
    if stop <= trainer.update:
        raise ValueError('Requested stopping update must exceed the checkpoint update.')
    history = out/'history.csv'
    fields = ['update', 'loss', 'denoising', 'aggregation', 'gradient_norm', 'learning_rate', 'validation_rmse']
    if history.exists():
        with history.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        if rows and int(rows[-1]['update']) != trainer.update:
            raise ValueError('History and resume checkpoint have different final updates; use a fresh output directory.')
    if resume and trainer.best_online is not None:
        if not history.exists() and (out/'best.pt').exists() and out.resolve() != Path(resume).resolve().parent:
            raise FileExistsError('Resume destination contains an unrelated best.pt; choose an empty directory.')
        trainer.save_best(out/'best.pt')
    with history.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        while trainer.update < stop:
            case = training[int(trainer.rng.integers(len(training)))]
            row = trainer.step(case)
            scheduled = trainer.update % training_config['validation_interval_updates'] == 0 or trainer.update == training_config['max_updates']
            if scheduled:
                score = trainer.validate(validation)
                row['validation_rmse'] = score
                # Strict comparison makes the earliest update win exact ties.
                if trainer.select(score):
                    trainer.save_best(out/'best.pt')
                print(f'Update {trainer.update}: loss={row["loss"]:.5g}, validation RMSE={score:.5g}', flush=True)
            writer.writerow(row)
            handle.flush()
            if scheduled or trainer.update == stop:
                trainer.save(out/'latest.pt', keys)
    return trainer
