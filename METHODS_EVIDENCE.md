# Methods evidence register

This file is the literature basis for the formal protocol in `full_experiment_config.yaml`.
The rule is:

1. **Direct literature basis** — the paper directly supports the method/definition being used.
2. **Literature-informed preregistration** — literature supports the principle, but the exact numerical value is a study-specific preregistered choice.
3. **Study-specific preregistration** — no paper is claimed to establish the exact value; the value is justified by leakage control, resource limits, or development-only sensitivity analysis.

The formal manuscript/report must not turn category 2 or 3 into a claim that the literature established that exact numerical setting.

## Evidence map

| Design element | Formal setting | Evidence class | Literature basis / interpretation |
|---|---|---|---|
| EGNN architecture | E(n)-equivariant message passing on 3D residue graphs | Direct literature basis | [R1] introduces E(n)-equivariant GNNs with rotation/translation/reflection/permutation equivariance. |
| Antibody-antigen interface label | Cross-partner heavy-atom distance <=5 Å | Direct literature basis | [R3] uses a 5 Å atom-distance cutoff for antigen-antibody interfaces; CAPRI contact definitions also use 5 Å atom contacts [R4]. |
| Dunbrack rotamer model | Backbone-dependent Dunbrack 2010 library; chi1..chiN | Direct literature basis | [R2] defines backbone-dependent rotamer probabilities, means and variances as functions of phi/psi. |
| Fixed-backbone side-chain optimization | Discrete rotamer selection with pairwise energies | Direct literature basis | Classical side-chain positioning/GMEC literature establishes the combinatorial rotamer-search formulation and its difficulty [R10-R12]. |
| QAOA | Gate-based hybrid approximate optimization | Direct literature basis | QAOA originates with Farhi et al. [R7]; protein/peptide quantum optimization provides domain precedent [R16-R18]. |
| QAOA depth | Primary p=2; development sensitivity p in {1,2,3} | Literature-informed preregistration | QAOA depth is a central hyperparameter and changes approximation quality/resource cost [R7,R9]. **No cited paper establishes p=2 as optimal for this nanobody QUBO.** p=2 is frozen only after development-only depth sensitivity. |
| CVaR QAOA objective | Primary CVaR alpha=0.1; alpha sensitivity 0.05..1.0 | Literature-informed preregistration | CVaR as a variational quantum optimization objective is supported by [R8]. **alpha=0.1 is not treated as a universal literature optimum.** |
| Simulated annealing baseline | SA included as classical stochastic optimizer | Direct literature basis | Classical simulated annealing is the canonical stochastic optimization method introduced by Kirkpatrick et al. [R13]. |
| FASPR baseline | External fixed-backbone side-chain packing baseline | Direct literature basis | FASPR is a peer-reviewed mature protein side-chain packing method [R14]. |
| Amber ff14SB | All-atom validation / calibration force field | Direct literature basis | ff14SB protein backbone and side-chain parameters are described in [R15]. |
| OpenMM | All-atom energy/relaxation engine | Direct literature basis | OpenMM 7 is described and benchmarked in [R19]. |
| Foldseek | Structure-similarity search feeding frozen clusters | Direct literature basis | Foldseek is a peer-reviewed fast protein structure search method [R5]. |
| Foldseek/TM-score cluster threshold | min_score=0.50 | Literature-informed preregistration | Foldseek supports the structure-search method [R5], but **0.50 is a study-specific frozen threshold**, not claimed as a Foldseek paper default/optimum for this task. |
| Antigen sequence isolation | identity <0.30 with >=0.70 coverage | Literature-informed preregistration | Sequence identity around 20-35% is the classical sequence-comparison twilight zone [R6]. The exact 0.30 identity and 0.70 coverage gates are conservative preregistered study rules. |
| VHH / CDR-H3 isolation | VHH <0.80; CDR-H3 <0.50 | Study-specific preregistration | Used to reduce close antibody-family leakage. These exact thresholds are **not represented as universal antibody standards** unless a task-specific paper is added later. |
| Intra-chain graph radius | C-alpha <8 Å | Study-specific preregistration | Common geometric neighborhood construction motivates a radius graph, but this register does not claim a paper establishes 8 Å as optimal for this task. |
| Cross-partner graph edges | fixed KNN, k=3 | Study-specific preregistration | Chosen specifically to decouple input edge existence from the 5 Å heavy-atom label and prevent deterministic topology leakage. No paper is claimed to establish k=3 as optimal. |
| Active-site scaling | 4, 6, 8, 10 sites | Study-specific preregistration | Motivated by the exponential/combinatorial growth of rotamer search [R10-R12] and the current <=30-variable representation. The four values are experimental scaling levels, not literature standards. |
| Primary active-site size | 6 sites | Study-specific preregistration | Kept as the frozen confirmatory/all-atom size for comparability and compute control. Not a literature optimum. |
| <=30 QUBO variables | hard representation cap | Study-specific preregistration | Computational/resource cap of the present simulator benchmark. Not claimed as a biological or literature-defined threshold. |
| 3-6 retained rotamers/site | adaptive state truncation | Literature-informed preregistration | Rotamer discretization itself is literature based [R2,R10-R12]; the 3-6 retained-state cap is specific to this <=30-variable benchmark. |
| Matched-output / matched-time comparisons | separate budget analyses | Direct methodological basis | Modern quantum-optimization benchmarking emphasizes explicit metrics, suitable classical comparators and resource-aware comparisons [R20]. |
| Quantum-advantage wording | simulator-level relative performance only; no hardware quantum advantage claim | Direct methodological basis | Quantum-optimization reviews emphasize rigorous benchmarking and distinguish empirical algorithm comparisons from demonstrations of hardware quantum advantage [R20]. |
| CAPRI-style contact metrics | Fnat/contact at 5 Å; iRMSD/LRMSD definitions kept explicit | Direct literature basis | CAPRI uses 5 Å interpartner atom contacts and standard interface/ligand RMSD concepts [R4]. Project-specific DockQ-like fields remain explicitly distinguished from official DockQ. |
| Holm multiplicity adjustment | Holm correction | Direct literature basis | Holm's sequentially rejective procedure controls family-wise error [R21]. |
| Family/structure-cluster statistical unit | repeated PDBs aggregated within frozen family clusters | Literature-informed preregistration | Independence is enforced by the frozen homology/structure clustering protocol; exact minimum cluster counts are preregistered adequacy gates, not literature power guarantees. |

## Formal interpretation rules

- A citation justifies the **method or scientific principle** only to the extent stated above.
- An exact parameter is called “literature-based” only when the cited paper directly supports that exact definition/value in a sufficiently similar setting.
- Values such as `cross_partner_knn_k=3`, `p=2`, `CVaR alpha=0.1`, `active_sites=[4,6,8,10]`, `primary_active_sites=6`, and the 30-variable cap remain preregistered study choices unless task-specific evidence is added.
- Validation/test data must never be used to choose these values after the freeze.
- The current QAOA results are classical exact-subspace simulations with finite-shot objectives; they can support quantum-classical **algorithmic relative-performance** statements, not a hardware quantum-advantage or quantum-speedup claim.

## References

**[R1]** Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). *E(n) Equivariant Graph Neural Networks*. ICML, PMLR 139, 9323-9332.

**[R2]** Shapovalov, M. V., & Dunbrack, R. L. Jr. (2011). A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions. *Structure*, 19(6), 844-858. https://doi.org/10.1016/j.str.2011.03.019

**[R3]** Ramaraj, T., Angel, T., Dratz, E. A., Jesaitis, A. J., & Mumey, B. (2012). Antigen-antibody interface properties: Composition, residue interactions, and features of 53 non-redundant structures. *Biochimica et Biophysica Acta*, 1824(3), 520-532. https://doi.org/10.1016/j.bbapap.2011.12.007

**[R4]** Lensink, M. F. et al. (2016). Prediction of homoprotein and heteroprotein complexes by protein docking and template-based modeling: A CASP-CAPRI experiment. *Proteins*, 84(S1), 323-348. https://doi.org/10.1002/prot.25007

**[R5]** van Kempen, M. et al. (2024). Fast and accurate protein structure search with Foldseek. *Nature Biotechnology*, 42, 243-246. https://doi.org/10.1038/s41587-023-01773-0

**[R6]** Rost, B. (1999). Twilight zone of protein sequence alignments. *Protein Engineering*, 12(2), 85-94. https://doi.org/10.1093/protein/12.2.85

**[R7]** Farhi, E., Goldstone, J., & Gutmann, S. (2014). *A Quantum Approximate Optimization Algorithm*. arXiv:1411.4028.

**[R8]** Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I., & Woerner, S. (2020). Improving Variational Quantum Optimization using CVaR. *Quantum*, 4, 256. https://doi.org/10.22331/q-2020-04-20-256

**[R9]** Pan, Y., Tong, Y., & Yang, Y. (2022). Automatic depth optimization for a quantum approximate optimization algorithm. *Physical Review A*, 105, 032433. https://doi.org/10.1103/PhysRevA.105.032433

**[R10]** Desmet, J., De Maeyer, M., Hazes, B., & Lasters, I. (1992). The dead-end elimination theorem and its use in protein side-chain positioning. *Nature*, 356, 539-542. https://doi.org/10.1038/356539a0

**[R11]** Hong, E.-J., Lippow, S. M., Tidor, B., & Lozano-Perez, T. (2009). Rotamer optimization for protein design through MAP estimation and problem-size reduction. *Journal of Computational Chemistry*, 30(12), 1923-1945. https://doi.org/10.1002/jcc.21188

**[R12]** Hallen, M. A., Keedy, D. A., & Donald, B. R. (2013). Dead-end elimination with perturbations (DEEPer): a provable protein design algorithm with continuous sidechain and backbone flexibility. *Proteins*, 81, 18-39. https://doi.org/10.1002/prot.24150

**[R13]** Kirkpatrick, S., Gelatt, C. D. Jr., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680. https://doi.org/10.1126/science.220.4598.671

**[R14]** Huang, X., Pearce, R., & Zhang, Y. (2020). FASPR: an open-source tool for fast and accurate protein side-chain packing. *Bioinformatics*, 36(12), 3758-3765. https://doi.org/10.1093/bioinformatics/btaa234

**[R15]** Maier, J. A. et al. (2015). ff14SB: Improving the Accuracy of Protein Side Chain and Backbone Parameters from ff99SB. *Journal of Chemical Theory and Computation*, 11(8), 3696-3713. https://doi.org/10.1021/acs.jctc.5b00255

**[R16]** Perdomo-Ortiz, A. et al. (2012). Finding low-energy conformations of lattice protein models by quantum annealing. *Scientific Reports*, 2, 571. https://doi.org/10.1038/srep00571

**[R17]** Khatami, M. H., Mendes, U. C., Wiebe, N., & Kim, P. M. (2023). Gate-based quantum computing for protein design. *PLoS Computational Biology*, 19(4), e1011033. https://doi.org/10.1371/journal.pcbi.1011033

**[R18]** Boulebnane, S. et al. (2023). Peptide conformational sampling using the Quantum Approximate Optimization Algorithm. *npj Quantum Information*, 9, 70. https://doi.org/10.1038/s41534-023-00733-5

**[R19]** Eastman, P. et al. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLoS Computational Biology*, 13(7), e1005659. https://doi.org/10.1371/journal.pcbi.1005659

**[R20]** Abbas, A. et al. (2024). Challenges and opportunities in quantum optimization. *Nature Reviews Physics*, 6, 718-735. https://doi.org/10.1038/s42254-024-00770-9

**[R21]** Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
