"""Regression tests for CVaR mass integration, bounded multistart optimization,
honest convergence reporting, and the finite-shot measurement ledger."""
import unittest
import numpy as np
from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler, lower_tail_cvar, finite_shot_cvar
from nanoqc.quantum.instance import QuantumOptimizationInstance


class RobustQAOATests(unittest.TestCase):
    def test_sampler_constructs_from_quantum_instance(self):
        instance=QuantumOptimizationInstance(
            Q=np.zeros((4,4)),constant_offset=0.0,
            physical_self=np.array([0.,2.,0.,5.]),
            physical_pair=np.zeros((4,4)),
            site_to_variables={0:[0,1],1:[2,3]},
            ising_h=np.zeros(4),ising_J=np.zeros((4,4)),ising_offset=0.0,
        )
        sampler=XYMixerQAOASampler.from_instance(
            instance,p=2,simulation_mode="subspace",seed=42)
        self.assertEqual(sampler.num_variables,instance.num_qubits)
        self.assertEqual(sampler.feasible_configuration_count,instance.feasible_configuration_count)
        np.testing.assert_array_equal(sampler.physical_self,instance.physical_self)

    def test_fractional_quantile(self):
        e=np.array([10.,0.,2.]); p=np.array([.5,.2,.3])
        self.assertAlmostEqual(lower_tail_cvar(e,p,.4),1.)
        self.assertAlmostEqual(lower_tail_cvar(e,p,1.),float(e@p))
        with self.assertRaises(ValueError):
            lower_tail_cvar(e,p,0.)

    def test_finite_shot_cvar_top_k_mean(self):
        # cvar_alpha=0.4 over 5 shots -> ceil(0.4*5)=2 lowest shots, plain mean.
        e = np.array([5., 1., 3., 2., 4.])
        self.assertAlmostEqual(finite_shot_cvar(e, .4), 1.5)
        # cvar_alpha=1 -> mean of every shot.
        self.assertAlmostEqual(finite_shot_cvar(e, 1.), float(e.mean()))
        # Non-integer alpha*N rounds the tail count up (ceil), not down.
        e2 = np.arange(10, dtype=float)
        self.assertAlmostEqual(finite_shot_cvar(e2, .15), 0.5)  # ceil(1.5)=2 -> mean(0,1)
        with self.assertRaises(ValueError):
            finite_shot_cvar(e, 0.)
        with self.assertRaises(ValueError):
            finite_shot_cvar(np.array([]), .5)
        with self.assertRaises(ValueError):
            finite_shot_cvar(np.array([1., np.nan]), .5)

    def test_budget_reproducible_and_physical_mean(self):
        def run():
            sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
                p=2,simulation_mode='subspace',seed=42)
            result=sampler.optimize_robust(max_evals=31,restarts=3,cvar_alpha=.2)
            self.assertLessEqual(result.evaluations,31)
            self.assertEqual(result.evaluations,len(result.history))
            self.assertEqual(result.evaluations,1+sum(r['evaluations'] for r in result.restart_records))
            self.assertLessEqual(result.objective_value,result.history[0]+1e-12)
            self.assertAlmostEqual(result.energy,sampler.expected_energy(np.r_[result.gammas,result.betas]))
            np.testing.assert_array_equal(sampler.physical_self,[0.,2.,0.,5.])
            return result
        a,b=run(),run()
        np.testing.assert_allclose(a.gammas,b.gammas)
        self.assertEqual(a.history,b.history)

    def test_insufficient_budget_rejected(self):
        sampler=XYMixerQAOASampler([0.,1.],np.zeros((2,2)),{0:[0,1]},simulation_mode='subspace')
        with self.assertRaises(ValueError):
            sampler.optimize_robust(max_evals=10,restarts=4)

    def test_default_finite_shot_ledger(self):
        sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
            p=2,simulation_mode='subspace',seed=7)
        result=sampler.optimize_robust()  # eval_shots=500, cvar_alpha=0.1, restarts=4, max_evals=90
        self.assertEqual(result.eval_shots,500)
        self.assertEqual(result.nfev,result.evaluations)
        self.assertEqual(result.nfev,len(result.history))
        self.assertEqual(result.total_opt_shots,result.nfev*500)
        self.assertEqual(result.shot_ledger['nfev'],result.nfev)
        self.assertEqual(result.shot_ledger['eval_shots'],500)
        self.assertEqual(result.shot_ledger['total_opt_shots'],result.total_opt_shots)
        self.assertEqual(result.shot_ledger['cvar_alpha'],0.1)
        self.assertLessEqual(len(result.history),90)

    def test_hard_shot_budget_cutoff_across_restarts(self):
        sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
            p=2,simulation_mode='subspace',seed=11)
        # Minimum feasible budget for 4 restarts at p=2: 1+4*(2*2+2)=25.
        result=sampler.optimize_robust(max_evals=25,restarts=4,eval_shots=50)
        self.assertLessEqual(result.nfev,25)
        self.assertEqual(result.total_opt_shots,result.nfev*50)
        # A best point is always exported, even when every restart's COBYLA
        # reports success=False because its own maxiter budget ran out.
        self.assertTrue(np.isfinite(result.energy))
        self.assertEqual(result.gammas.shape,(2,))
        self.assertEqual(result.betas.shape,(2,))
        np.testing.assert_array_equal(result.gamma_best,result.gammas)
        np.testing.assert_array_equal(result.beta_best,result.betas)
        self.assertEqual(result.best_mean_energy,result.energy)

    def test_eval_shots_is_mandatory_and_strictly_positive(self):
        # No analytic-expectation escape hatch: every objective evaluation
        # must be finite-shot; eval_shots is a plain positive int, not Optional.
        sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
            p=2,simulation_mode='subspace',seed=3)
        with self.assertRaises(ValueError):
            sampler.optimize_robust(max_evals=25,restarts=4,eval_shots=None)
        with self.assertRaises(ValueError):
            sampler.optimize_robust(max_evals=25,restarts=4,eval_shots=0)

    def test_honest_termination_reporting_under_tight_budget(self):
        # A 6-evaluation-per-restart quota on a noisy 4-dim finite-shot CVaR
        # objective is essentially guaranteed not to satisfy COBYLA's own
        # catol convergence test, so this must be reported honestly as
        # budget-exhausted, never silently upgraded to "converged".
        sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
            p=2,simulation_mode='subspace',seed=17)
        result=sampler.optimize_robust(max_evals=25,restarts=4,eval_shots=30)
        self.assertEqual(result.nfev,25)  # every restart's quota is exactly exhausted
        self.assertEqual(result.termination_reason,"max_evaluations_reached")
        self.assertFalse(result.optimizer_success)
        self.assertFalse(result.success)  # legacy alias must agree with optimizer_success
        for record in result.restart_records:
            self.assertIn(record['termination_reason'],
                ("max_evaluations_reached","converged","stopped_early_without_convergence",
                 "restart_raised_exception"))
        self.assertIn("max_evaluations_reached",result.message)

    def test_solve_with_measurement_ledger_returns_dict(self):
        sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
            p=2,simulation_mode='subspace',seed=5)
        ground=sampler.enumerate_ground_states()
        solved=sampler.solve_with_measurement_ledger(
            max_evals=31,restarts=3,eval_shots=100,output_shots=777,ground_state=ground)
        self.assertIsInstance(solved,dict)
        ledger=solved['measurement_ledger']
        self.assertEqual(ledger['eval_shots'],100)
        self.assertEqual(ledger['output_shots'],777)
        self.assertEqual(ledger['total_measurement_shots'],ledger['total_opt_shots']+777)
        self.assertEqual(solved['quantum_sample'].shots,777)
        self.assertEqual(solved['quantum_sample'].legal_rate,1.0)
        # New diagnostics are present both nested and mirrored at the top level.
        self.assertEqual(solved['bitstring_entropy'],solved['quantum_sample'].bitstring_entropy)
        self.assertEqual(solved['low_energy_fraction'],solved['quantum_sample'].low_energy_fraction)
        self.assertEqual(solved['ground_state_hit'],solved['quantum_sample'].ground_state_hit)
        self.assertIn('optimizer_success',solved)
        self.assertIn('termination_reason',solved)
        np.testing.assert_array_equal(solved['gamma_best'],solved['gammas'])
        np.testing.assert_array_equal(solved['beta_best'],solved['betas'])
        self.assertEqual(solved['best_mean_energy'],solved['energy'])

    def test_sample_output_diagnostics(self):
        sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
            p=2,simulation_mode='subspace',seed=9)
        ground=sampler.enumerate_ground_states()
        # Zero angles -> uniform distribution over the 4 feasible states; sample it directly.
        sampled=sampler.sample(np.zeros(4),shots=1000,ground_state=ground,energy_window=2.0)
        self.assertEqual(sampled.shots,1000)
        self.assertEqual(sampled.legal_rate,1.0)
        self.assertGreaterEqual(sampled.bitstring_entropy,0.0)
        self.assertLessEqual(sampled.bitstring_entropy,float(np.log(1000)))
        self.assertGreaterEqual(sampled.low_energy_fraction,sampled.ground_state_success_probability)
        self.assertEqual(sampled.ground_state_hit,sampled.ground_state_success_probability>0.0)
        with self.assertRaises(ValueError):
            sampler.sample(np.zeros(4),shots=10,energy_window=-1.0)

    def test_mixed_three_to_six_state_registers(self):
        groups={0:[0,1,2],1:[3,4,5,6],2:[7,8,9,10,11]}
        n=sum(len(g) for g in groups.values())
        physical_self=np.linspace(0.0,2.0,n)
        pair=np.zeros((n,n))
        sampler=XYMixerQAOASampler(
            physical_self,pair,groups,p=2,simulation_mode='subspace',seed=23)
        self.assertEqual(sampler.num_variables,12)
        self.assertEqual(sampler.feasible_configuration_count,3*4*5)
        ground=sampler.enumerate_ground_states()
        self.assertEqual(ground.configuration_count,60)
        self.assertTrue(all(sampler.is_legal(state) for state in ground.states))
        sampled=sampler.sample(np.zeros(4),shots=300,ground_state=ground,sample_seed=29)
        self.assertEqual(sampled.legal_rate,1.0)
        self.assertTrue(all(sampler.is_legal(state) for state in sampled.counts))
        annealed=sampler.simulated_annealing(num_reads=40,site_passes=5,seed=31,ground_state=ground)
        self.assertTrue(sampler.is_legal(annealed.best_state))
        self.assertTrue(all(sampler.is_legal(state) for state in annealed.counts))

    def test_six_state_register_is_accepted(self):
        groups={0:[0,1,2,3,4,5]}
        sampler=XYMixerQAOASampler(np.zeros(6),np.zeros((6,6)),groups,simulation_mode='subspace')
        self.assertEqual(sampler.feasible_configuration_count,6)
        self.assertTrue(all(sampler.is_legal(state) for state in sampler.feasible_energy_map()))
    def test_invalid_arguments_rejected(self):
        sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
            p=2,simulation_mode='subspace',seed=1)
        with self.assertRaises(ValueError):
            sampler.optimize_robust(eval_shots=-5)
        with self.assertRaises(ValueError):
            sampler.optimize_robust(eval_shots=1.5)
        with self.assertRaises(ValueError):
            sampler.optimize_robust(cvar_alpha=0.)
        with self.assertRaises(ValueError):
            sampler.optimize_robust(cvar_alpha=1.5)
        with self.assertRaises(ValueError):
            sampler.optimize_robust(restarts=0)
        with self.assertRaises(ValueError):
            sampler.optimize_robust(objective='bogus')
        with self.assertRaises(ValueError):
            sampler.optimize_robust(parameter_scale='bogus')
        with self.assertRaises(ValueError):
            sampler.solve_with_measurement_ledger(output_shots=0)


if __name__=='__main__':
    unittest.main()
