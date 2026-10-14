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
from recodif.training import Trainer, learning_rate, model_from_state

ROOT = Path(__file__).resolve().parents[1]


def fixture(device='cpu'):
    grid = Grid(np.linspace(-3, 1, 3), np.linspace(-3, 1, 3))
    cfg = json.loads((ROOT/'configs/model.json').read_text())
    cfg.update(width=16, heads=2, blocks=1, terminal_time=.8, minimum_training_time=.03)
    settings = json.loads((ROOT/'configs/training.json').read_text())
    settings.update(max_updates=8, effective_batch_size=8, lr_warmup_updates=0,
                    denoising_only_warmup_updates=1, teacher_ema=.8, validation_interval_updates=8,
                    validation_draw_count=3)
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
            cases.append(Case(str(index), site, 'validation', 1., case.grid, case.evidence, heldout=heldout))
        with patch('recodif.training.posterior_draws', return_value=np.ones((3, 4))):
            assert_allclose(trainer.validate(cases), 3.5)
            cases[0].role = 'test'
            with self.assertRaisesRegex(ValueError, 'validation cases only'):
                trainer.validate(cases)

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
