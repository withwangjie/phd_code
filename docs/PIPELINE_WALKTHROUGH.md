# 研究流程详解

本文按执行顺序讲清正式流水线每个阶段做什么、依据什么规则、产出什么。
参数取自 `configs/full_experiment_config.yaml`，方法学依据见
`docs/METHODS_EVIDENCE.md`（引用编号 `[R*]`），偏离预注册的改动见
`docs/PROTOCOL_AMENDMENTS.md`（编号 `A*`），产物验收标准见
`docs/RESULTS_CONTRACT.md`。

## 研究目标

在纳米抗体–抗原界面上用 XY 混合器 QAOA 重建侧链构象，检验的是**量子算法本身
把概率质量集中到基态的能力**，不是它比经典求解器更快（A7）。

主要终点是逐实例训练的 QAOA 相对均匀可行采样的 log10 精确基态振幅放大倍数，
以及该放大随可行构象数的标度斜率。在 ≤10 位点的规模上模拟退火 100% 命中基态，
胜负事先已定，所以量子对经典的一切比较都是次要终点。

## 阶段依赖

`src/nanoqc/pipeline/run_full_experiment.py` 的 `STAGE_PREREQUISITES` 是唯一来源：

```
env_check
  └─ smoke_check
       └─ data_audit
            └─ queue_freeze  ★冻结点
                 ├─ egnn_train
                 │    ├─ energy_calibration
                 │    │    ├─ method_sensitivity
                 │    │    │    └─ qc_benchmark
                 │    │    └─ quantum_exploration
                 │    └─ structure_experiment
                 └─ (structure_experiment 亦依赖 queue_freeze)
                      └─ external_validation
                           └─ statistics
                                └─ final_report
```

续跑时沿这张图向上回溯校验新鲜度，不只看直接前置——否则 `data_audit` 变了而
`queue_freeze` 未重跑的情况，在校验 `egnn_train` 时会被漏掉。

---

## 〇、一次性准备（正式运行之外）

产物被正式运行按哈希绑定。

| 步骤 | 命令 | 产物 |
|---|---|---|
| 独立审计 | `./scripts/prepare_external_vhh.sh audit` | `data/external_vhh/prep/study_pdb_ids.txt` |
| 补齐分辨率 | `python -m nanoqc.data.fetch_entry_resolution` | `data/entry_resolution.tsv` |
| Foldseek 聚类表 | 正式运行在 `queue_freeze` 自动构建；需安装 Foldseek 或设置 `QP_FOLDSEEK` | `<run>/independence/foldseek_pairs.tsv` + manifest |

### Foldseek 表的构建规则

1. 每个 PDB 只保留**非抗体链**，长度 ≥20 aa。抗体链与 VHH 共享 Ig 折叠，
   留下它们会把几乎所有复合物连成一个单链接分量（A9）。
2. 抗体链按顺序识别：SNAC/SAbDab 序列标注（观测 ≥70 残基、≥70% 标注长度、
   为子序列）→ SAbDab 链 ID → Ig 可变域检测（有 ANARCI 用 ANARCI，
   否则用 C23/W41/C104/`[WF]G.G` 保守锚点）。标注过的抗原链一律保留。
3. 穷尽式全对全搜索：`--exhaustive-search 1 -e 10`，`--max-seqs` = 条目数。
4. 链对打分 `mintmscore = min(qtmscore q→t, qtmscore t→q)`，即按较长链归一化
   （A11）。只在一个方向检出的链对不连边。PDB 对取其链对的最大值。
5. 无抗原链的 PDB 写一条自比对行（TM=1.0），只能成为单例。
6. **全集只含经来源核验、行级 QC 合格的正式 VHH 候选**，且来源与图准入所用的
   优先来源一致（A16）。被拒的结构不能充当单链接抗原分量之间的桥。
7. manifest 记录每条链的角色/方法/长度、自比对列表、pair 表与原始搜索的哈希。

`--reuse-raw` 对已有的 `foldseek_raw.m8` 重新打分，不重跑搜索；
`--run-dir <run>` 把表绑定到某次运行的全集，避免覆盖校验失败。

---

## 一、env_check

检查 Python 依赖、Dunbrack 转子库（`data/rotamer/ALL.bbdep.rotamers.lib`，
缺失即失败）、FASPR、Phenix clashscore 的可用性。

## 二、smoke_check

小规模冒烟，跑通消融与重建两条链路：1 个靶点（`4s10`）、5 位点、
1 个种子（42）、12 次评估、20 个输出样本。读本次运行的数据集，
不回落到历史路径。

---

## 数据来源角色（A16）

正式角色在任何质量或结果被查看之前固定，跨来源重复的 PDB 按
`SNAC > SAbDab > RCSB` 的确定性优先级取舍：

| 来源 | 角色 | 说明 |
|---|---|---|
| `snac_db` | **primary** | SNAC-DB 逐复合物的整理标注 [R34]，正式评估总体 |
| `sabdab_vhh` | **auxiliary** | 读自身 H/L/抗原元数据，拒绝 L 链/scFv/非多肽抗原，须在组装体内确认唯一 VHH，记录 CDR 标注方法，拒绝未标注的额外 Ig 可变域 [R33] |
| `train_rcsb` | **audit-only** | 仅参与审计，不进入正式训练 |

RCSB 之所以只做审计：组装体和分辨率检查证明的是坐标质量，不是纳米抗体身份。
把通用蛋白最强界面的第一条链当作 VHH 会改变学习任务本身。它要重回正式训练，
需要先补上可独立审计的严格 VHH 与抗原角色标注，并另立修订。

优先级只依据来源身份和 PDB ID，是无结果的决策：优先来源事后未通过质量门槛时，
**不会**回落到低优先级的表示，所以准入无法靠查看下游质量或模型结果来挽救。

## 三、data_audit

### 子集归类

| 目录 | 子集 |
|---|---|
| `rcsb_non_redundant_dataset/` | `train_rcsb` |
| `sabdab_all_sd_h_structures/` | `sabdab_vhh` |
| `SNAC-DataBase/curated_structures/nb_complexes*` | `snac_db` |
| `benchmark5.5/` | `test_db55` |
| `external_vhh/` | 跳过（流水线暂存目录，A12） |

### 逐结构检查

| 项 | 规则 |
|---|---|
| 解析 | 首模型，至少一个正占有率、坐标有限的氨基酸重原子残基 |
| 组装体 | `sabdab_vhh`/`train_rcsb` 取第一个作者确定的生物学组装体（软件确定为回落）；文件名为 `*_assembly<N>` 的视为已组装，不再施加变换（A14）；无组装体标注的不对称单元排除 |
| 主链 | 已观测残基 N/CA/C/O 齐全 |
| 界面 | 跨链重原子 <5 Å [R3,R4]，最强链对接触残基数 ≥15 |
| VHH | SNAC 非 TCR、唯一 H 链、无 L 链、序列覆盖 ≥70% 且与标注一致，端部扩展合计 ≤20 aa |
| CDR-H3 | 取 SNAC 的 IMGT `Region_Split_VH.cdr3`，不套用未经确认的残基编号 |
| 质量 | 分辨率 ≤3.0 Å；界面侧链完整；界面无 altloc；界面最低占有率 ≥0.90 |

### 分辨率来源

按顺序回落，逐行记 `resolution_source`：结构文件 → SNAC curation summary →
SAbDab 汇总表 → RCSB `entry_resolution.tsv`。一个字段列出多个值时取**最差**
（最大）值，因为门槛是上界（A12、A15）。任何来源都查不到的条目（NMR 等）
按 `unknown_resolution` 排除，阈值不变。

实际用到的元数据文件连同 SHA-256 列在审计报告里，流水线工作目录下的文件
不参与 [R48]。

### 产物

`data_audit_details.jsonl` / `.csv`、`data_audit_report.md`、
`data_audit_db55_pairs.json`

---

## 四、queue_freeze ★预注册冻结点

训练之前一次性定死全部数据决策，此后不可更改。这是防止事后调参的核心环节。

### 4.1 冻结聚类全集

收集本次审计的全部四字符 PDB ID → `audit/cluster_universe.txt`。
校验 pair 表覆盖全集，缺一个即失败。

### 4.2 冻结结构聚类映射

pair 表按 `min_score: 0.50`、`score_semantics: mintmscore` 单链接聚类
→ `independence/pdb_family_clusters.json`。TM-score 0.50 有公开的
折叠/拓扑显著性依据 [R27]。

已存在的映射必须经来源校验（绑定到当前 pair 表与阈值），对不上则拒绝——
防止换了表却复用旧聚类。

### 4.3 构建图数据集

**图定义 v1.10**：每张图消费的原始结构都绑定到数据审计记录的 SHA-256，
构建前重校字节，使审计与构建不可能悄悄跨越不同的结构快照（A16）。
v1.8 / v1.9 的旧图无法在此语义下续跑。

| 元素 | 规则 | 依据 |
|---|---|---|
| 节点 | 残基 | |
| 链内边 | CA 距离 <8 Å | [R22] |
| 跨伙伴边 | KNN，k=3 | 研究特定的防泄漏选择，不声称文献最优 |
| 界面标签 | 跨伙伴重原子 <5 Å | [R3,R4] |
| 最小界面 | 15 个残基 | |
| 抗原定义 | 组装体内任一 CA/CB 距 VHH CDR1–3 的 CA/CB <7.5 Å 的链 | [R33] |

**hard test 选取**（`--no-cap`，处理全部合格簇）：

1. 候选 = `snac_db` 中单条 CDR-H3 且长度 ≥16 aa、不与 DB5.5 重叠的复合物；
2. 按 CDR-H3 loop 50% 身份聚类 [R23]；
3. 按冻结种子打乱簇顺序，每簇取一个代表；
4. 图级再审一遍分层同源性。

**训练集**：扣除与 hard test 在任一层同源的全部复合物。

**产物**：`graphs/{train,test_snac_hard,test_db55}/*.pt`、
`graph_manifest.json` / `.csv`、`excluded_samples.csv`（逐样本多原因）、
`graph_dataset_delivery_report.md`

### 4.4 切出抗原折叠 holdout

SAbDab 给不出真正的外部集：2429 条单域条目里 2424 条已在本研究的审计范围内，
剩下 4 条只构成 2 个独立组，p 值下限 0.5，任何效应都无法声称（A10）。
因此改为从训练集切出整个分层组件作为 holdout。

1. 对训练图求分层隔离的连通分量；
2. 组件折号 = 成员文件名排序拼接后的 SHA-256 前 8 字节 mod 5；
3. 内部验证折固定为 0；holdout 取 `fold`，取值 `"1"` 或 `"1,2"`，
   规则是**最小的几个非验证折号**，绝不挑组件最多的那一折（A13）；
4. 组件 ≥50 张图**且**大于池子的 1/5 → 固定进训练集，不参与分折（A11）。
   否则这样一个组件落进某折就会成为该折的绝大部分；
5. 切分前后被固定的组件集合必须一致，否则失败；
6. 组件须**含有 SNAC primary 图**才有资格成为 holdout（A17）；
7. 整个选中组件从 `graphs/train` 移出。其中每个 PDB 取**一张确定性的 SNAC 图**
   进 `graphs/holdout` 并被评分；SAbDab 成员图与同 PDB 的重复 SNAC 图移入
   `graphs/holdout_quarantine`，既不训练也不评分；
8. 每个被评分的靶点按精确的审计 `source_id` 与审计时的 SHA-256 解析原始结构，
   不靠 PDB ID——跨来源或同 PDB 多个整理复合物会造成绑定歧义（A17）；
9. **门控**：holdout 组件数 < `min_components`（10）即失败；
   剩余训练组件 < `min_train_components`（20）亦失败。

分层组件仍由 SNAC primary 与 SAbDab auxiliary **联合**构建，防泄漏屏障不变；
SAbDab 只是不成为共同的正式评估总体。每 PDB 一个靶点避免伪重复，
也与下游"一个 PDB 一份原始结构"的契约一致。holdout manifest 为 schema v2。

移除整个组件不改变其余组件的折号，所以内部验证划分与不切 holdout 时完全一致。

### 4.5 冻结验证靶点队列

- 排除历史开发靶点 `4s10`、`8yvo`、`9gcn`，永不再进入验证队列；
- 选取顺序 `seeded_random`，绝不按结构大小升序；
- `target_count: 0` 表示全部合格靶点，无隐藏上限；
- **合格判定**只看两条：序列/家族隔离，以及至少 6 个独立的 Dunbrack/Amber
  兼容可动 VHH 残基。用 `--eligibility-only`，**不做任何残基排序**。

此时 EGNN 尚未训练，所以合格判定只能用不依赖检查点的廉价策略（`contact`）；
若被误配成 `egnn` 会被拒绝。真正的 EGNN 排序推迟到 `structure_experiment`，
届时通过 `--pdb-allowlist-file` 显式复现同一集合，不靠重新推导。

### 门控产物

`independence/cluster_adequacy.json`、`independence/antigen_fold_holdout.json`。
二者都是无结果门控（A5）：只看数据构成，不看任何结果。

---

## 五、egnn_train

E(n) 等变 GNN [R1] 输出残基级界面概率。

| 参数 | 值 |
|---|---|
| 隐藏维度 / 层数 | 32 / 4 |
| dropout / 坐标缩放 | 0.10 / 0.10 |
| 学习率 / 权重衰减 | 1e-3 / 1e-5 |
| 最大轮数 / 早停耐心 | 50 / 5 |
| 梯度裁剪 | 5.0 |
| 混合精度 | 开 |

训练/内部验证划分沿用 `queue_freeze` 的分层组件加名称哈希折（验证折 0），
不做随机 90/10。

`seed_replicates: 2`：在同一划分上另训 2 个模型（派生训练种子），报告
ROC-AUC 离散度与 Active-site Jaccard 稳定性 [R44]。这是**仅开发用**的
种子敏感性（A6），正式模型始终只有主检查点一个。

## 六、energy_calibration

把粗粒度能量拟合到 Amber14 全原子能量差上，**只用训练集复合物**，拟合后冻结。

- 岭回归 `alpha: 1.0`；位点选择用 `egnn`（与下游正式分布一致）；
  每复合物 64 个赋值；6 位点；半径 6 Å；
- 接受门槛：训练复合物 ≥20、训练组 ≥20、交叉验证 ≥5 折且**按家族分组**、
  RMSE ≤10 kcal、MAE ≤7.5 kcal、R² ≥0.20、Spearman ≥0.40、生成失败率 ≤10%；
- 验证/测试行永不被拟合器接受。

粗粒度力场参数（截断 8 Å、软核 0.5 Å、硬核比例 0.72、硬球罚 25.0、
LJ 斥力上限 50.0、LJ 吸引上限 5.0、库仑上限 20.0、介电基值 4.0、
介电斜率 2.0、热能 0.593 kcal）是模型参数而非结合亲和力常数，
全部写进 QUBO 元数据。

## 七、method_sensitivity

**仅开发/训练图**，主协议保持冻结。

- 输入 `graphs/train`，最多 20 个靶点，6 位点，300 输出，重复 3 次；
- 超参敏感性：深度 1/2/3 × 评估次数 90/180/300 × 评估采样 200/500/1000 ×
  CVaR α 0.05/0.10/0.25/0.50/1.0；
- 转子分辨率敏感性：4–5 位点 × 每位点 3/4/5/6 个状态，八种条件共用同一
  训练子集与重复种子。这是表示/求解器敏感性，**不是量子优势检验**。

## 八、qc_benchmark

输入 `graphs/test_snac_hard`，`max_targets: 0`（全部合格图，无静默取子集），
`max_failure_fraction: 0.0`（失败即停）。

消融扫描：5 种位点选择（`egnn`/`contact`/`distance`/`cdr`/`random`）×
输出预算曲线（10/30/100/300/1000）× 目标函数（mean/CVaR）×
起点数（1/4），位点数 4/6/8/10，半径 6 Å，每种情形 10 个独立重复身份。

每个求解器的开销字段（`solver_seconds`、`oracle_seconds`、
`single_state_energy_queries`）全部显式记录，**等输出绝不等同于等预算**。
SA 基线 100 轮 [R13]，贪心 50 轮，能量窗口 2.0。

## 九、quantum_exploration

探索性、描述性，**不用于选择冻结的主协议**（A7）。

- 深度 p ∈ {1,2,3,4,6}，每个深度给 `20 × 2p` 次目标评估，
  把电路深度与优化预算分离 [R7,R45]；6 位点，每深度 3 个重复身份；
- 参数迁移：在训练图（最多 40 个，最少 10 个实例）上取优化后、
  按尺度归一化的角度的逐分量中位数，未经训练直接用到 hard set [R46,R47]。

## 十、structure_experiment

在**已冻结**的开发队列与验证队列上做真实全原子实验。

### 位点选择

`(1-w)·EGNN + w·exp(-d_Ag/6Å)`，`w = 0.25`。
contact / 最近距离 / CDR / 随机是等预算消融基线，绝不静默替换主策略。

### 问题编码

one-hot 转子寄存器，6 位点 × 3 个 chi1 井 = 3⁶ = 729 个可行构象。
W 态初始化（局部 one-hot W 态之积）+ 局部 XY 混合器 [R28,R30]，
搜索始终留在可行子空间内，不需要惩罚项压制非法解。

### 冻结的量子协议

唯一来源是 `quantum_protocol.primary`；验证/测试结果永不用于回调这一块。

| 参数 | 值 |
|---|---|
| 深度 | 2 |
| 最大评估次数 | 90 |
| 目标函数 | CVaR，α = 0.10 [R8] |
| 起点数 | 4 |
| 评估采样 / 输出采样 | 500 / 1000 |
| 参数尺度 | max_coefficient |

### 结构协议

5 个重复身份（42–46），多 chi 扰动 40°–120°，真空溶剂模型
（GBn2 仅开发队列），松弛 200 次迭代、候选 100 次、loop 100 次，
`robust_qaoa: true`。

保存扰动前、仅松弛、离散、stage1、stage2 五种结构及其指标，
支撑最终报告里的"能量下降 vs RMSD 上升"配对分析。
VAL/LEU 甲基不按对称处理（A3）。

## 十一、external_validation

`required: true`，输入缺失即失败。

`graph_dir` 与 `source_structure_dir` 留空 → 用本次运行在 `queue_freeze`
切出的抗原折叠 holdout；填一对目录 → 评分真正的外部 VHH 集。
**两种情况的独立性审计完全相同。**

独立性审计每次运行重新生成为
`external_validation/external_vhh_independence_manifest.json`，
把每张图绑定到原始结构文件核验残基与 CA 坐标。`repeats: 10`，
`min_clusters: 10`。

同时跑成熟结构基线：FASPR 重建、Phenix clashscore，超时 1800 秒。

## 十二、statistics

重采样 10000 次。**串行把关（A4）**，不是扁平的 Holm 家族 [R41,R42]。

1. **主要族**（A7）：
   1. 6 位点处 log10 精确基态振幅放大对 0 的双侧聚类符号翻转检验；
   2. 该放大对 log10 可行构象数的组内标度斜率（固定 p=2）。
2. 主要族通过后才检验次要族——包括降级后的 `log10_qts99`（主基线 SA）
   与排除训练开销的 `log10_qts99_execution` [R37,R45]。
3. 主要位点选择固定 `egnn`，半径 6.0，结构终点 `final_rmsd`，
   结构对比 `qaoa_vs_sa`。
4. 充分性门槛：`min_qc_clusters`、`min_scaling_clusters`、
   `min_primary_clusters`、`min_rq5_clusters` 均为 10。这是门槛，不是效能保证。
5. G 个独立聚类时最小可达 p 值为 2/2^G，G ≥ 6 才可能到 p < 0.05 [R43]。
6. RQ5 若不可估计，按预先指定的"不可估计结局"报告，
   排除在 Holm 之外（A2）[R39]。
7. 统计**必须复用** `queue_freeze` 冻结的那份聚类映射；配置里故意不提供
   `statistics.cluster_map` 覆盖项，防止划分与统计的聚类漂移。
8. 时间预算模式 `outputs` 与 `time` 都报，超时容忍 10%。

## 十三、final_report

输出 `FINAL_RESEARCH_REPORT.md`。

运行结束做结果契约审计：每阶段的必需产物必须存在、大小相符，
≤64 MiB 的重算 SHA-256 并比对。**子进程返回 0 不等于阶段完成。**

---

## 贯穿全程的约束

### 分层同源隔离

任一条件触发即判为同源，连通分量不得跨任何划分：

| 层 | 阈值 | 依据 |
|---|---|---|
| VHH 全链身份 | ≥0.80 | [R24] |
| CDR-H3 loop 身份 | ≥0.50 | [R23] |
| 抗原全链身份 | ≥0.30 且长度覆盖 ≥0.70 | [R25] |
| 共享冻结结构簇 | mintmscore ≥0.50 | [R5,R27] |

身份计算用全局 Needleman–Wunsch、BLOSUM62、gap-open 10、gap-extend 1。
角色未经标注锚定的复合物额外做一次**交换伙伴**的比对，使放在抗原槽位的
纳米抗体仍按 VHH 阈值受检，跨角色同源不被漏掉。

独立性声明仅限该聚类输入及其相似性定义，不外推为对所有远缘同源关系的
绝对排除。

### 冻结与哈希

`master_seed: 4050350448` 于运行前独立抽取。代码、配置、pair 表、
聚类映射、元数据文件全部记 SHA-256，续跑时重算比对，不一致即拒绝。
子种子由种子流按 `(靶点, 重复身份)` 派生，不直接复用配置里的数字。

### 修订留痕

任何偏离预注册的改动写入 `docs/PROTOCOL_AMENDMENTS.md`，
须注明修订前查看了哪些结果。A9–A15 标"不适用"，因为改动针对的都是数据构成，
当时没有任何结果存在；A16、A17 的该字段**待研究者本人填写**。

## 运行命令

```bash
# 冻结前先验：只跑到 queue_freeze 停下，确认数据构成站得住
./scripts/deploy_launch.sh --stop-after queue_freeze

# 全流程
./scripts/deploy_launch.sh
```

`queue_freeze` 之后检查：

```bash
RUN=$(ls -td runs/experiments_full_run_* | head -1)
cat $RUN/independence/cluster_adequacy.json
cat $RUN/independence/antigen_fold_holdout.json
```
