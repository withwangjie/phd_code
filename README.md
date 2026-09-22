# phd_code

## Research scope

This repository benchmarks constrained quantum-classical optimization for
nanobody (VHH)-antigen interface side-chain reconstruction under a known
complex pose/backbone. It is a retrospective side-chain recovery benchmark,
not blind docking, de novo complex prediction, or quantitative binding-affinity
prediction.

## Formal research pipeline

1. **Leakage-controlled data construction**
   - Interface labels: cross-partner heavy-atom contact < 5 A.
   - Graph edges: intra-chain CA radius < 8 A plus fixed cross-partner KNN.
   - EGNN train/validation split: bilateral full-chain VHH + antigen
     40%-identity connected components; no random 90/10 split.

2. **Antigen-conditioned Active-site selection**
   - SE(3)-equivariant EGNN provides residue-level interface probabilities.
   - Formal EGNN ranking uses
     `(1-w) * EGNN + w * exp(-d_Ag/6A)`, with `w=0.25` by default.
   - Contact, CDR and random strategies are explicit ablation baselines.

3. **Adaptive side-chain state construction**
   - Formal coarse and all-atom protocols both use residue-flexibility-aware
     raw chi1 sub-rotamer pools of 6/9/12 states.
   - Candidate pre-screening uses local environment / antigen-conditioned
     interaction scoring (coarse model) or Amber14 single-candidate energy
     (all-atom validation).
   - 3--6 states per Active residue are retained under a global <=30-variable
     budget.
   - Formal benchmark default: 6 Active residues; supported range 5--8.

4. **Constrained discrete optimization**
   - One-hot residue registers are encoded as QUBO/Ising variables.
   - XY-mixer QAOA preserves local Hamming weight and therefore feasibility.
   - Classical baselines include exact feasible-state enumeration and
     simulated annealing.
   - Mean-energy and finite-shot CVaR QAOA objectives are compared.

5. **Structure-level validation**
   - Solver assignments are reconstructed as side-chain conformations.
   - OpenMM constrained relaxation evaluates whether discrete energy gains
     persist after continuous structural refinement.
   - Structural metrics include Active side-chain RMSD, Fnat, interface RMSD,
     ligand RMSD, clash measures, and related trajectory metrics.

## Central research questions

- Does EGNN retain useful interface-selection ability after homology isolation
  and topology-leakage control?
- Does adaptive 3--6-state modeling improve coverage over fixed low-state
  discretization?
- How do XY-QAOA and classical solvers differ in energy gap, low-energy
  coverage, sampling diversity, and budget efficiency?
- Does CVaR emphasize useful low-energy conformations better than mean-energy
  optimization?
- Most importantly, do solver-level discrete energy gains propagate to
  all-atom structural improvements?

## Interpretation limits

- Coarse antigen interaction scores are not binding free energies.
- Contact number is a geometry baseline, not an affinity estimator.
- Current all-atom experiments remain native-backbone-conditioned recovery
  controls.
- No quantum advantage claim should be made without matched-budget evidence.
- Smoke checks and legacy explicit `chi1_angles` overrides are engineering or
  ablation paths and are not the formal main protocol.

## Main entry points

- `run_full_experiment.py`: end-to-end orchestrator.
- `build_final_pyg_dataset.py`: audited graph construction.
- `train_egnn_pruning.py`: leakage-controlled EGNN training.
- `model_egnn_pruning.py`: interface scoring and Active-site selection.
- `subgraph_to_qubo.py`: coarse/all-atom adaptive side-chain QUBO builders.
- `qaoa_interface_sampler.py`: XY-mixer QAOA.
- `batch_benchmark_hard_set.py`: matched quantum/classical benchmark.
- `run_real_complex_pilot.py`: all-atom retrospective recovery experiment.
- `evaluate_complex_metrics.py`: structure-level evaluation.
