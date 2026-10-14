import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
import torch
from recodif.data import Case, load_manifest
from recodif.evidence import Evidence
from recodif.model import SharedFlux
from recodif.partitions import Grid
from recodif.reference import lognormal_prior, posterior_references
from recodif.sampling import posterior_draws
from recodif.training import Trainer, learning_rate, model_from_state, train

ROOT = Path(__file__).resolve().parents[1]


def fixture(device='cpu'):
    grid = Grid(np.linspace(-3, 1, 3), np.linspace(-3, 1, 3))
    cfg = json.loads((ROOT/'configs/model.json').read_text())
    cfg.update(width=16, heads=2, blocks=1, terminal_time=.8, minimum_training_time=.03)
    settings = json.loads((ROOT/'configs/training.json').read_text())
    settings.update(max_updates=8, effective_batch_size=8, lr_warmup_updates=0,
                    denoising_only_warmup_updates=1, teacher_ema=.8, validation_interval_updates=8,
                    validation_draw_count=3, reference_draw_count_per_evidence=64)
    schedule = dict(grids=[[1, 1], [2, 2]], times=[.3], reverse_step_size=.2,
                    quadrature_tolerance=1e-8, potential_tolerance=2e-4)
    torch.manual_seed(42)
    model = SharedFlux(grid, cfg).to(device)
    evidence = Evidence(np.array([-.3, .2, .4]), np.array([[-1., -.4, .2, .3], [.2, .7, .1, .4], [.1, .5, .2, .6]]), np.eye(3)*.04)
    draws = np.random.default_rng(3).lognormal(-2, .4, (64, 4))
    case = Case('a', 'site_a', 'train', 1., grid, evidence, draws)
    return Trainer(model, settings, schedule, 19), case


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_warmup_gradients_teacher_direction(self):
        trainer, case = fixture()
        first = trainer.step(case)
        self.assertIsNone(trainer.teacher)
        self.assertEqual(first['aggregation'], 0.)
        old = [p.detach().clone() for p in trainer.model.parameters()]
        second = trainer.step(case)
        self.assertGreaterEqual(second['aggregation'], 0.)
        self.assertTrue(any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in trainer.model.observation_encoder.parameters()))
        for before, teacher, online in zip(old, trainer.teacher.parameters(), trainer.model.parameters()):
            torch.testing.assert_close(teacher, .8*before+.2*online)
            self.assertIsNone(teacher.grad)
            self.assertFalse(teacher.requires_grad)

    def test_independent_repeat_count_reaches_network(self):
        trainer, case = fixture()
        evidence = Evidence([1.], [[.1, .2, .3, .4]], [[1.]])
        repeated = Evidence([1., 1.], [[.1, .2, .3, .4], [.1, .2, .3, .4]], np.eye(2))
        first = trainer.model.encode_evidence(evidence)
        second = trainer.model.encode_evidence(repeated)
        self.assertFalse(torch.allclose(first[0, 0], second[0, 0]))
        partition = case.grid.uniform_partition((2, 2))
        mass = np.full((1, 4), .2)
        h1 = trainer.model.residual(mass, partition, .4, first)
        h2 = trainer.model.residual(mass, partition, .4, second)
        self.assertFalse(torch.equal(h1, h2))

    def test_checkpoint_exact_cpu_continuation_and_online_load(self):
        trainer, case = fixture()
        trainer.step(case)
        trainer.step(case)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.pt'
            trainer.save(path, [('a', 'site_a', 'train')])
            expected = trainer.step(case)
            resumed, keys = Trainer.resume(path)
            actual = resumed.step(case)
            self.assertEqual(actual, expected)
            self.assertEqual(keys, [('a', 'site_a', 'train')])
            for p, q in zip(trainer.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            for p, q in zip(trainer.teacher.parameters(), resumed.teacher.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            state = torch.load(path, weights_only=True)
            loaded = model_from_state(state)
            for name, parameter in loaded.state_dict().items():
                torch.testing.assert_close(parameter, state['online'][name])

    def test_fixed_cir_batch_learning(self):
        trainer, case = fixture()
        batch = trainer.batch(case)
        trainer.model.train()
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.002, weight_decay=0.)
        initial = float(trainer.losses(case.evidence, batch, False)[0].detach())
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = trainer.losses(case.evidence, batch, False)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        final = float(trainer.losses(case.evidence, batch, False)[0].detach())
        self.assertLess(final, initial*.35, f'Fixed-batch denoising objective: {initial} -> {final}')

    def test_inference_finishes_on_reference_grid(self):
        trainer, case = fixture()
        trainer.step(case)
        trainer.step(case)
        result = posterior_draws(trainer.model, case.evidence, trainer.schedule, 4, seed=23)
        self.assertEqual(result.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(result)) and np.all(result >= 0))
        self.assertTrue(trainer.model.training)
        assert_allclose(result, posterior_draws(trainer.model, case.evidence, trainer.schedule, 4, seed=23), rtol=0, atol=0)

    def test_scheduler_endpoints(self):
        trainer, _ = fixture()
        config = dict(trainer.config, lr_warmup_updates=2)
        self.assertEqual(learning_rate(0, config), config['learning_rate']/2)
        self.assertEqual(learning_rate(1, config), config['learning_rate'])
        self.assertEqual(learning_rate(7, config), config['minimum_learning_rate'])

    def test_validation_weights_sites_equally(self):
        trainer, case = fixture()
        cases = []
        for index, (site, y) in enumerate((('A', 1.), ('A', 3.), ('B', 5.))):
            heldout = (np.array([y]), np.zeros((1, 4)), np.eye(1), np.zeros((1, 3)))
            cases.append(Case(str(index), site, 'validation', 1., case.grid, case.evidence, heldout=heldout,
                              well_id=site, heldout_train_ids=np.array(['train_1'])))
        with patch('recodif.training.posterior_draws', return_value=np.ones((3, 4))):
            assert_allclose(trainer.validate(cases), 3.5)
            cases[0].role = 'test'
            with self.assertRaisesRegex(ValueError, 'validation cases only'):
                trainer.validate(cases)

    def test_reference_budget_is_not_silently_ignored(self):
        trainer, case = fixture()
        trainer.config['reference_draw_count_per_evidence'] = 512
        with self.assertRaisesRegex(ValueError, 'exactly 512 reference draws'):
            trainer.batch(case)

    def test_selected_online_weights_survive_standalone_resume(self):
        trainer, case = fixture()
        trainer.step(case)
        self.assertTrue(trainer.select(1.))
        selected = {k: v.clone() for k, v in trainer.best_online.items()}
        trainer.step(case)
        self.assertFalse(trainer.select(1.))  # Earliest update wins a tie.
        self.assertFalse(trainer.select(2.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'latest.pt'
            trainer.save(path)
            # No companion best.pt exists; latest must retain the selection itself.
            resumed, _ = Trainer.resume(path)
            resumed.step(case)
            resumed.save_best(Path(directory)/'selected.pt')
            state = torch.load(Path(directory)/'selected.pt', weights_only=True)
            self.assertEqual(state['update'], 1)
            self.assertEqual(resumed.update, 3)
            for key, expected in selected.items():
                torch.testing.assert_close(state['online'][key], expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'resume training from latest.pt'):
                Trainer.resume(Path(directory)/'selected.pt')
            legacy = torch.load(path, weights_only=True)
            del legacy['selection_metric']
            torch.save(legacy, Path(directory)/'legacy.pt')
            with self.assertRaisesRegex(ValueError, 'earlier flattened validation RMSE'):
                Trainer.resume(Path(directory)/'legacy.pt')

    def test_training_resume_into_new_directory_keeps_earlier_selection(self):
        trainer, case = fixture()
        validation = Case('v', 'validation_site', 'validation', 1., case.grid, case.evidence,
                          heldout=(np.array([1.]), np.zeros((1, 4)), np.eye(1), np.zeros((1, 3))),
                          well_id='v_well', heldout_train_ids=np.array(['v_train']))
        settings = dict(trainer.config, max_updates=4, validation_interval_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/'first', Path(directory)/'second'
            with patch.object(Trainer, 'validate', side_effect=[1., 2.]):
                train([case, validation], trainer.model.config, settings, trainer.schedule,
                      first, seed=19, stop_after=2)
            selected = torch.load(first/'best.pt', weights_only=True)
            with patch.object(Trainer, 'validate', side_effect=[2., 3.]):
                resumed = train([case, validation], trainer.model.config, settings, trainer.schedule,
                                second, seed=19, resume=first/'latest.pt')
            inherited = torch.load(second/'best.pt', weights_only=True)
            self.assertEqual(resumed.update, 4)
            self.assertEqual(inherited['update'], 1)
            for name in selected['online']:
                torch.testing.assert_close(inherited['online'][name], selected['online'][name], rtol=0, atol=0)

    def test_microbatch_limits_reverse_and_allocation_evaluation(self):
        trainer, case = fixture()
        batches = []
        # Known constant field isolates batching from floating point network differences.
        def field(mass, partition, time, encoded):
            batches.append(len(mass))
            return torch.full((len(mass), partition.size), -2., dtype=torch.float32)
        with patch.object(trainer.model, 'residual', side_effect=field):
            small = posterior_draws(trainer.model, case.evidence, trainer.schedule, 7, seed=76, sample_microbatch=3)
            self.assertLessEqual(max(batches), 3)
            batches.clear()
            large = posterior_draws(trainer.model, case.evidence, trainer.schedule, 7, seed=76, sample_microbatch=64)
            self.assertGreater(max(batches), 3)
        assert_allclose(small, large, rtol=0, atol=0)

    def test_invalid_schedules_are_rejected_before_sampling(self):
        trainer, case = fixture()
        for override in ({'schedule': 'adaptive'}, {'times': [.9]}, {'reverse_step_size': np.nan},
                         {'split_order': 'random'}, {'grids': [[1.5, 1], [2, 2]]}):
            with self.assertRaises(ValueError):
                posterior_draws(trainer.model, case.evidence, dict(trainer.schedule, **override), 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_loss_backward_update_and_reload(self):
        trainer, case = fixture('cuda')
        first = trainer.step(case)
        second = trainer.step(case)
        self.assertTrue(np.isfinite(first['loss']+second['loss']))
        self.assertTrue(all(p.device.type == 'cuda' for p in trainer.model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cuda.pt'
            trainer.save(path)
            resumed, _ = Trainer.resume(path, 'cuda')
            self.assertTrue(np.isfinite(resumed.step(case)['loss']))


class DataReferenceTests(unittest.TestCase):
    def test_chain_balanced_training_subset(self):
        from recodif.reference import select_reference_draws
        chains = np.arange(4*512*2, dtype=float).reshape(4, 512, 2)
        original = chains.copy()
        selected, indices = select_reference_draws(chains, count=512, seed=91)
        self.assertEqual(selected.shape, (512, 2))
        assert_allclose(np.bincount(indices[:, 0]), [128]*4)
        self.assertEqual(len(set(map(tuple, indices))), 512)
        assert_allclose(selected, chains[indices[:, 0], indices[:, 1]])
        assert_allclose(chains, original)
        assert_allclose(select_reference_draws(chains, 512, 91)[0], selected)
        with self.assertRaises(ValueError):
            select_reference_draws(chains, count=2049)

    def test_multiple_masks_of_one_record_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path/'case.npz', y=[0.], K=np.zeros((1, 4)), covariance=[[1.]],
                     log_t1_edges=[-3., -1., 1.], log_t2_edges=[-3., -1., 1.],
                     y_holdout=[1., 2.], K_holdout=np.zeros((2, 4)), covariance_hh=np.eye(2),
                     heldout_train_ids=['train_A', 'train_B'])
            entries = [dict(record_id='depth_01', site_id='site', well_id='well', role='validation',
                            mask_id=str(i), path='case.npz') for i in range(2)]
            (path/'manifest.json').write_text(json.dumps({'cases': entries}))
            cases = load_manifest(path/'manifest.json')
            self.assertEqual(len(cases), 2)
            self.assertEqual([c.mask_id for c in cases], ['0', '1'])
            entries[1]['mask_id'] = '0'
            (path/'manifest.json').write_text(json.dumps({'cases': entries}))
            with self.assertRaisesRegex(ValueError, 'Repeated site/well/record/mask'):
                load_manifest(path/'manifest.json')
    def test_reference_diagnostics_on_independent_chains(self):
        from recodif.reference import reference_diagnostics
        grid = Grid([-4., -2., -1., 0., 1.], [-4., -2., -1., 0., 1.])
        samples = np.random.default_rng(927).gamma(2., 1., (4, 1000, 16))
        report = reference_diagnostics(samples, grid)
        self.assertTrue(report['criteria_met'])
        self.assertEqual(len(report), 18)

    def test_prior_mean_and_reference_sampler(self):
        trainer, case = fixture()
        prior = json.loads((ROOT/'configs/protocol.json').read_text())['prior']
        mean, chol = lognormal_prior(case.grid, prior)
        expectation = case.grid.areas*np.exp(mean+.5*np.diag(chol @ chol.T))
        assert_allclose(expectation.sum(), 1.)
        # Zero kernel gives exactly the lognormal prior, a known posterior.
        evidence = Evidence([0.], np.zeros((1, 4)), [[1.]])
        draws = posterior_references(evidence, case.grid, prior, draws_per_chain=1500, burnin=50, seed=16)
        self.assertEqual(draws.shape, (4, 1500, 4))
        assert_allclose(draws.mean((0, 1)), expectation, rtol=.12)

    def test_site_overlap_is_rejected_without_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps({'cases': [dict(record_id='a', site_id='same', role='test', path='unused.npz'),
                               dict(record_id='b', site_id='same', role='recalibration', path='unused.npz')]}))
            with self.assertRaisesRegex(ValueError, 'site cannot span'):
                load_manifest(path)


if __name__ == '__main__':
    unittest.main()
