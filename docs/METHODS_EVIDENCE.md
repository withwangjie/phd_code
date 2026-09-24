# Methods evidence register

This file is the literature basis for the formal protocol in `full_experiment_config.yaml`.
The rule is:

1. **Direct literature basis** — the paper directly supports the method/definition being used.
2. **Literature-informed preregistration** — literature supports the principle, but the exact numerical value is a study-specific preregistered choice.
3. **Study-specific preregistration** — no paper is claimed to establish the exact value; the value is justified by leakage control, resource limits, or development-only sensitivity analysis.

The formal manuscript/report must not turn category 2 or 3 into a claim that the literature established that exact numerical setting.

## Evidence map

### Data, labels and leakage control

| Design element | Formal setting | Evidence class | Literature basis / interpretation |
|---|---|---|---|
| Complex definition (antigen chains) | `sabdab_vhh`/`train_rcsb`: first author-determined biological assembly (software-determined fallback; entries without assembly annotation excluded); antigen = assembly chains with any CA/CB within 7.5 A of a VHH CDR1-3 CA/CB (whole chain if CDRs cannot be mapped); further VHH copies are not antigen. `snac_db`: SNAC-DB assembly-curated complexes as distributed | Direct literature basis | SAbDab assigns antigens by CA/CB proximity (7.5 A) to CDR residues within the author-defined biological assembly [R33]; SNAC-DB, the source of `snac_db`, extracts complexes with biological-assembly logic [R34]. The asymmetric unit alone is not a biological complex: crystal-packing neighbours can appear as false partners and symmetry-generated partners are missing [R35,R36]. Annotated assemblies are themselves imperfect (AVIDbase needed the ASU for 61%, the annotated assembly for 38% and symmetry mates for 1% of nanobody complexes [R35]), so the contact rule is applied inside the assembly and the chosen assembly, contact basis and dropped chains are recorded per graph. |
| Antibody-antigen interface label | Cross-partner heavy-atom distance <=5 Å | Direct literature basis | [R3] uses a 5 Å atom-distance cutoff for antigen-antibody interfaces; CAPRI contact definitions also use 5 Å atom contacts [R4]. |
| Antigen sequence isolation | identity <0.30 with >=0.70 coverage | Literature-supported empirical setting | [R6] establishes the classical low-identity/twilight-zone context. More directly, PepNN [R25] removes test proteins with >30% identity at 70% coverage to train/validation proteins. The present antigen rule adopts that exact anti-leakage combination as a conservative precedent, without claiming it is the only valid threshold for antibody antigens. |
| VHH full-chain isolation | identity <0.80 | Literature-supported antibody/nanobody precedent | H3-OPT [R24] constructs a non-redundant antibody dataset including nanobodies using sequence identity <0.8 before train/validation/test splitting. The present VHH rule uses this published non-redundancy level as a conservative precedent, not as a universal VHH standard. |
| CDR-H3 isolation | identity <0.50 | Direct antibody-design benchmark precedent | DiffAb [R23] and subsequent antibody-design benchmarks cluster/split SAbDab antibodies at 50% CDR-H3 sequence identity. The present split adopts the same threshold to reduce CDR-H3 leakage. |
| Foldseek | Structure-similarity search feeding frozen clusters; antigen chains only (>=20 residues), exhaustive all-versus-all | Direct literature basis for the method; study-specific input scope | Foldseek is a peer-reviewed fast protein structure search method [R5]. Antibody chains share the Ig fold, so they are removed before the search; otherwise single linkage would join almost all complexes. Antibody-side leakage is controlled by the VHH and CDR-H3 sequence thresholds (PROTOCOL_AMENDMENTS.md A9). |
| Foldseek/TM-score cluster threshold | min_score=0.50 | Direct literature basis for the structural-similarity cutoff | Foldseek supplies the structure-search method [R5]. Independently, [R27] shows a rapid same-fold/topology transition around TM-score=0.5 and proposes 0.5 as a rough quantitative fold/topology cutoff. The pair table must have a verified protein-length-normalized `qtmscore` or `ttmscore` header; default `fident` and alignment-normalized `alntmscore` are rejected. |
| CAPRI-style contact metrics | Fnat/contact at 5 Å; iRMSD/LRMSD definitions kept explicit | Direct literature basis | CAPRI uses 5 Å interpartner atom contacts and standard interface/ligand RMSD concepts [R4]. Project-specific DockQ-like fields remain explicitly distinguished from official DockQ. |

### Interface model and Active-site selection

| Design element | Formal setting | Evidence class | Literature basis / interpretation |
|---|---|---|---|
| EGNN architecture | E(n)-equivariant message passing on 3D residue graphs | Direct literature basis | [R1] introduces E(n)-equivariant GNNs with rotation/translation/reflection/permutation equivariance. |
| Intra-chain graph radius | C-alpha <8 Å | Literature-supported protein-graph setting | GLINTER [R22] represents residues by C-alpha atoms and explicitly constructs residue-graph edges using a distance cutoff such as 8 Å. This directly supports 8 Å as a published protein-graph setting, while not claiming it is universally optimal for this EGNN task. |
| Cross-partner graph edges | fixed KNN, k=3 | Study-specific preregistration | Chosen specifically to decouple input edge existence from the 5 Å heavy-atom label and prevent deterministic topology leakage. No paper is claimed to establish k=3 as optimal. |
| EGNN seed variance | development-only replicate training; ROC-AUC spread and Active-site Jaccard stability | Direct methodological basis | Random-seed variance materially changes learned-benchmark results and should be reported [R44] (PROTOCOL_AMENDMENTS.md A6). |
| Active-site scaling | 4, 6, 8, 10 sites | Study-specific preregistration | Motivated by the exponential/combinatorial growth of rotamer search [R10-R12] and the current <=30-variable representation. The four values are experimental scaling levels, not literature standards. |
| Primary active-site size | 6 sites | Study-specific preregistration | Kept as the frozen confirmatory/all-atom size for comparability and compute control. Not a literature optimum. |

### Side-chain states and energy model

| Design element | Formal setting | Evidence class | Literature basis / interpretation |
|---|---|---|---|
| Dunbrack rotamer model | Backbone-dependent Dunbrack 2010 library; chi1..chiN | Direct literature basis | [R2] defines backbone-dependent rotamer probabilities, means and variances as functions of phi/psi. |
| 3-6 retained rotamers/site | adaptive state truncation | Literature-supported adaptive-state design | Rotamer discretization is supported by [R2,R10-R12]. Jumper et al. [R26] further use residue-dependent 1, 3, or 6 coarse rotamer states and explicitly discuss the accuracy/cost tradeoff. This supports adaptive residue-dependent state counts and an upper level of six; this study's minimum of three and global <=30-variable allocation remain preregistered implementation constraints. |
| <=30 QUBO variables | hard representation cap | Study-specific preregistration | Computational/resource cap of the present simulator benchmark. Not claimed as a biological or literature-defined threshold. |
| Fixed-backbone side-chain optimization | Discrete rotamer selection with pairwise energies | Direct literature basis | Classical side-chain positioning/GMEC literature establishes the combinatorial rotamer-search formulation and its difficulty [R10-R12]. |
| Coarse environment scope | Every antigen and fixed-VHH residue of the source complex; only the 8 A coarse atom-pair cutoff limits it | Direct literature basis for the scope, study-specific coarse parameters | Standard fixed-backbone packers score each rotamer's one-body energy against all neighbouring fixed residues, with neighbours defined by the potential's own atom-pair interaction range rather than an arbitrary pre-truncated region [R31,R32]. The environment radius therefore no longer limits any energy term; an exact CA pre-filter (cutoff + maximum pseudo-atom offset) removes no scored pair. |
| Amber ff14SB | All-atom validation / calibration force field | Direct literature basis | ff14SB protein backbone and side-chain parameters are described in [R15]. |
| OpenMM | All-atom energy/relaxation engine | Direct literature basis | OpenMM 7 is described and benchmarked in [R19]. |

### Quantum algorithm

| Design element | Formal setting | Evidence class | Literature basis / interpretation |
|---|---|---|---|
| QAOA | Gate-based hybrid approximate optimization | Direct literature basis | QAOA originates with Farhi et al. [R7]; protein/peptide quantum optimization provides domain precedent [R16-R18]. |
| Feasibility-preserving XY mixer | Local one-hot registers remain inside the feasible Hamming-weight-one subspace | Direct methodological basis | Hadfield et al. [R28] establish alternating-operator QAOA for hard-constrained feasible subspaces. Wang et al. [R30] directly analyze XY mixers for one-hot encodings and W-state initialization. The exact local all-pairs ordering used here remains a study-specific implementation detail. |
| QAOA initial state | Product of equal-amplitude local W states, one excitation per rotamer register | Direct methodological basis | Wang et al. [R30] study XY mixers under one-hot encoding and report the generalized W state as a natural/high-performing feasible initialization. The project uses an exact local W state in each independent one-hot register. |
| QAOA depth | Primary p=2; development sensitivity p in {1,2,3} | Literature-informed preregistration | QAOA depth is a central hyperparameter and changes approximation quality/resource cost [R7,R9]. **No cited paper establishes p=2 as optimal for this nanobody QUBO.** p=2 is frozen only after development-only depth sensitivity. |
| CVaR QAOA objective | Primary CVaR alpha=0.1; alpha sensitivity 0.05..1.0 | Literature-supported empirical setting | [R8] supports CVaR for variational quantum optimization, explicitly evaluates alpha=0.10/0.25/1.0, and recommends alpha in approximately [0.1,0.25] as a good empirical range. alpha=0.1 is therefore literature-supported, while this study still preregisters it and retains the broader alpha sensitivity analysis. |
| QAOA depth x optimizer budget (exploratory) | p in {1,2,3,4,6}, 20 evaluations per parameter | Literature-informed exploratory design | QAOA quality depends on depth [R7]; noiseless scaling studies use p up to 12 [R45]. Coupling the budget to the parameter count separates circuit depth from optimizer budget. Descriptive only. |
| QAOA parameter transfer (exploratory) | median scale-normalised optimal angles from training graphs, applied untrained | Direct literature basis | Optimal QAOA angles concentrate across typical instances [R46] and transfer between instances [R47]; fixed angles are used in recent QAOA scaling work [R45]. Fitted on the train split only. |
| Logical quantum-resource reporting | qubits, variational parameters, logical RZ/ZZ/XY counts, two-qubit gates and shots | Direct methodological basis for transparent resource reporting | Quantum-optimization benchmarking should report solution quality together with computational-resource use and clearly defined metrics [R20,R29]. The present counts are explicitly pre-transpilation logical counts; no hardware-native gate-count claim is made. |

### Endpoints and baselines

| Design element | Formal setting | Evidence class | Literature basis / interpretation |
|---|---|---|---|
| Primary quantum-intrinsic endpoint | log10 exact ground-state amplification of QAOA over uniform feasible sampling at 6 sites, and its scaling slope | Literature-informed preregistration | Success (ground-state) probability is a core metric for stochastic quantum optimizers [R29,R37]; comparing it with uniform sampling isolates the algorithm's own concentration effect. Exact subspace probabilities are simulator-level. Chosen because the study aims to explore the quantum algorithm (PROTOCOL_AMENDMENTS.md A7). |
| Quantum scaling analysis | primary predictor log10 feasible-state count; qubits and logical two-qubit gates are descriptive resource axes | Literature-informed preregistration | [R20,R29] motivate size-stratified, resource-aware quantum/classical benchmarking. The exact active-site levels and the choice of log10 feasible-state count as the single confirmatory predictor remain study-specific preregistration; qubit/gate axes are reported descriptively to avoid post-hoc multiplicity. |
| Resource-normalized QAOA-vs-classical endpoint (secondary since A7) | log10 queries-to-solution (99%), QAOA optimization shots charged; per-sample success probability by Jeffreys estimate | Direct literature basis for the metric; study-specific resource units | Time-to-solution R99 = t ln(0.01)/ln(1-p) is the standard benchmarking metric for stochastic optimizers and quantum heuristics [R37]; resource-aware comparison with explicit accounting is recommended [R20,R29]. The Jeffreys estimate is a recommended binomial-proportion estimator that stays finite at k=0 or k=n [R38]. One resource unit = one measurement shot or one single-state energy query (study-specific). Replaces best-of-N gap, which saturated (docs/PROTOCOL_AMENDMENTS.md A1). |
| Matched-output / matched-time comparisons | separate budget analyses | Direct methodological basis | Modern quantum-optimization benchmarking emphasizes explicit metrics, suitable classical comparators and resource-aware comparisons [R20,R29]. |
| Simulated annealing baseline | SA included as classical stochastic optimizer | Direct literature basis | Classical simulated annealing is the canonical stochastic optimization method introduced by Kirkpatrick et al. [R13]. |
| FASPR baseline | External fixed-backbone side-chain packing baseline | Direct literature basis | FASPR is a peer-reviewed mature protein side-chain packing method [R14]. |
| Quantum-advantage wording | simulator-level relative performance only; no hardware quantum advantage claim | Direct methodological basis | Quantum-optimization reviews emphasize rigorous benchmarking and distinguish empirical algorithm comparisons from demonstrations of hardware quantum advantage [R20]. |

### Statistics and protocol governance

| Design element | Formal setting | Evidence class | Literature basis / interpretation |
|---|---|---|---|
| Family/structure-cluster statistical unit | repeated PDBs aggregated within frozen family clusters | Literature-informed preregistration | Independence is enforced by the frozen homology/structure clustering protocol; exact minimum cluster counts are preregistered adequacy gates, not literature power guarantees. |
| Cluster adequacy before outcomes | independent-cluster minima checked at queue freeze | Literature-informed preregistration | Inference with few clusters is unreliable [R43]; the numerical minima remain preregistered adequacy gates, not power guarantees (PROTOCOL_AMENDMENTS.md A5). |
| Confirmatory multiplicity (coarse QC) | serial gatekeeping: primary family {primary QC contrast, primary scaling slope} Holm-adjusted alone; secondary effects gated behind it | Direct literature basis | Gatekeeping procedures test hierarchically ordered families while controlling FWER strongly [R41]; regulatory guidance recommends a pre-specified primary/secondary hierarchy with gatekeeping or fixed-sequence testing [R42] (PROTOCOL_AMENDMENTS.md A4). |
| Holm multiplicity adjustment | Holm correction within each gatekeeping family and in the structural two-test family | Direct literature basis | Holm's sequentially rejective procedure controls family-wise error [R21]. |
| Non-estimable RQ5 association | constant cluster-level difference reported as a named non-estimable outcome, excluded from Holm | Direct methodological basis | Statistical analysis plans should pre-specify handling of undefined estimates [R39] (docs/PROTOCOL_AMENDMENTS.md A2). |
| Protocol amendments | dated amendment log with inspection status | Direct methodological basis | Deviations from a preregistered plan must be disclosed transparently [R40]; see docs/PROTOCOL_AMENDMENTS.md. |

## Formal interpretation rules

- A citation justifies the **method or scientific principle** only to the extent stated above.
- An exact parameter is called “literature-based” only when the cited paper directly supports that exact definition/value in a sufficiently similar setting.
- Values that remain primarily study-specific are `cross_partner_knn_k=3`, `p=2`, `active_sites=[4,6,8,10]`, `primary_active_sites=6`, the 30-variable cap, and engineering compute budgets. CVaR alpha=0.1 is now literature-supported by [R8] as an empirical setting, but remains preregistered and sensitivity-tested for this specific nanobody QUBO.
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

**[R22]** Xie, Z., & Xu, J. (2022). Deep graph learning of inter-protein contacts. *Bioinformatics*, 38(4), 947-953. https://doi.org/10.1093/bioinformatics/btab761

**[R23]** Luo, S., Su, Y., Peng, X., Wang, S., Peng, J., & Ma, J. (2022). Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures. *Advances in Neural Information Processing Systems*, 35, 9754-9767.

**[R24]** Chen, H., Fan, X., Zhu, S., Pei, Y., Zhang, X., Zhang, X., Liu, L., Qian, F., & Tian, B. (2024). Accurate prediction of CDR-H3 loop structures of antibodies with deep learning. *eLife*, 12:RP91512. https://doi.org/10.7554/eLife.91512.4

**[R25]** Abdin, O., Nim, S., Wen, H., & Kim, P. M. (2022). PepNN: a deep attention model for the identification of peptide binding sites. *Communications Biology*, 5, 503. https://doi.org/10.1038/s42003-022-03445-2

**[R26]** Jumper, J. M., Faruk, N. F., Freed, K. F., & Sosnick, T. R. (2018). Accurate calculation of side chain packing and free energy with applications to protein molecular dynamics. *PLoS Computational Biology*, 14(12), e1006342. https://doi.org/10.1371/journal.pcbi.1006342

**[R27]** Xu, J., & Zhang, Y. (2010). How significant is a protein structure similarity with TM-score = 0.5? *Bioinformatics*, 26(7), 889-895. https://doi.org/10.1093/bioinformatics/btq066

**[R28]** Hadfield, S., Wang, Z., O'Gorman, B., Rieffel, E. G., Venturelli, D., & Biswas, R. (2019). From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz. *Algorithms*, 12(2), 34. https://doi.org/10.3390/a12020034

**[R29]** Koch, T., Bernal Neira, D. E., Chen, Y., et al. (2026). The Quantum Optimization Benchmarking Library. *Nature Computational Science*, 6, 653-671. https://doi.org/10.1038/s43588-026-00991-1

**[R30]** Wang, Z., Rubin, N. C., Dominy, J. M., & Rieffel, E. G. (2020). XY mixers: Analytical and numerical results for the quantum alternating operator ansatz. *Physical Review A*, 101(1), 012320. https://doi.org/10.1103/PhysRevA.101.012320

**[R31]** Krivov, G. G., Shapovalov, M. V., & Dunbrack, R. L. Jr. (2009). Improved prediction of protein side-chain conformations with SCWRL4. *Proteins*, 77(4), 778-795. https://doi.org/10.1002/prot.22488

**[R32]** Leaver-Fay, A., Tyka, M., Lewis, S. M., et al. (2011). ROSETTA3: an object-oriented software suite for the simulation and design of macromolecules. *Methods in Enzymology*, 487, 545-574. https://doi.org/10.1016/B978-0-12-381270-4.00019-6

**[R33]** Dunbar, J., Krawczyk, K., Leem, J., Baker, T., Fuchs, A., Georges, G., Shi, J., & Deane, C. M. (2014). SAbDab: the structural antibody database. *Nucleic Acids Research*, 42(D1), D1140-D1146. https://doi.org/10.1093/nar/gkt1043

**[R34]** Gupta, A. et al. (2026). SNAC-DB: An ML-ready database for antibody and NANOBODY® VHH-antigen complexes with expanded structural diversity and real-world benchmarking. *Protein Science*. https://doi.org/10.1002/pro.70655

**[R35]** Medved, T., Lah, J., Miličić, G., & Hadži, S. (2026). AVIDbase: A biologically accurate structural dataset of nanobody-antigen complexes. *Protein Science*, 35(9), e70773. https://doi.org/10.1002/pro.70773

**[R36]** Krissinel, E., & Henrick, K. (2007). Inference of macromolecular assemblies from crystalline state. *Journal of Molecular Biology*, 372(3), 774-797. https://doi.org/10.1016/j.jmb.2007.05.022

**[R37]** Rønnow, T. F., Wang, Z., Job, J., Boixo, S., Isakov, S. V., Wecker, D., Martinis, J. M., Lidar, D. A., & Troyer, M. (2014). Defining and detecting quantum speedup. *Science*, 345(6195), 420-424. https://doi.org/10.1126/science.1252319

**[R38]** Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101-133. https://doi.org/10.1214/ss/1009213286

**[R39]** Gamble, C., Krishan, A., Stocken, D., et al. (2017). Guidelines for the content of statistical analysis plans in clinical trials. *JAMA*, 318(23), 2337-2343. https://doi.org/10.1001/jama.2017.18556

**[R40]** Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences*, 115(11), 2600-2606. https://doi.org/10.1073/pnas.1708274114

**[R41]** Dmitrienko, A., & Tamhane, A. C. (2007). Gatekeeping procedures with clinical trial applications. *Pharmaceutical Statistics*, 6(3), 171-180. https://doi.org/10.1002/pst.291

**[R42]** U.S. Food and Drug Administration (2022). *Multiple Endpoints in Clinical Trials: Guidance for Industry*. https://www.fda.gov/media/162416/download

**[R43]** Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372. https://doi.org/10.3368/jhr.50.2.317

**[R44]** Bouthillier, X., Delaunay, P., Bronzi, M., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems*, 3, 747-769.

**[R45]** Shaydulin, R., Li, C., Chakrabarti, S., et al. (2024). Evidence of scaling advantage for the quantum approximate optimization algorithm on a classically intractable problem. *Science Advances*, 10(22), eadm6761. https://doi.org/10.1126/sciadv.adm6761

**[R46]** Brandão, F. G. S. L., Broughton, M., Farhi, E., Gutmann, S., & Neven, H. (2018). For fixed control parameters the quantum approximate optimization algorithm's objective function value concentrates for typical instances. arXiv:1812.04170.

**[R47]** Galda, A., Liu, X., Lykov, D., Alexeev, Y., & Safro, I. (2021). Transferability of optimal QAOA parameters between random graphs. *IEEE International Conference on Quantum Computing and Engineering (QCE)*. arXiv:2106.07531.
