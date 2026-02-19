# BenchCore Usage Example

## 1. Protein Evaluation 

### Usage

```python
from evaluation.evaluation_api import Evaluation
from omegaconf import DictConfig

config = DictConfig({...})
evaluator = Evaluation(config)

# run_protein evaluation
evaluator.run_protein_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv",
    ca_rmsd_threshold=2.0
)
```

### Pipeline script

```bash
python scripts/run_protein_pipeline.py \
    design_dir=/path/to/designs \
    inversefold=ProteinMPNN \
    refold=esmfold
```

## 2. Protein Binding Protein (PBP) Evaluation

### 使用方式

```python
evaluator.run_protein_binding_protein_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv"
)
```

### Pipeline 脚本

```bash
python scripts/run_pbp_pipeline.py \
    design_dir=/path/to/designs \
    inversefold=ProteinMPNN \
    refold=esmfold
```

## 3. Protein Binding Ligand (PBL) Evaluation

### 使用方式

```python
evaluator.run_protein_binding_ligand_evaluation(
    input_dir="/path/to/input",
    output_dir="/path/to/output",
    dist_cutoff=10.0,
    exhaustiveness=16
)
```

### Pipeline 脚本

```bash
python scripts/run_pbl_pipeline.py \
    design_dir=/path/to/designs \
    inversefold=LigandMPNN \
    refold=esmfold
```

## 4. Nucleotide (NUC) Evaluation

### 使用方式

```python
evaluator.run_nuc_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv"
)
```

### Pipeline 脚本

```bash
python scripts/run_nuc_pipeline.py \
    design_dir=/path/to/designs \
    inversefold=ODesign \
    refold=af3
```

## 5. Motif Scaffolding Evaluation (新增)

### 架构说明

Motif Scaffolding 使用 Generator + Evaluator 架构：
- **Generator 层** (`generators/`): 处理模型特定的数据转换
  - `PPIFlowGenerator`: 修复全ALA序列，结构比对
  - `RFD3Generator`: 解析JSON文件，提取contig信息
- **Evaluator 层** (`evaluation/motif_evaluator.py`): 模型无关的评估逻辑

### 使用方式 1: 通过 evaluation_api

```python
from evaluation.evaluation_api import Evaluation
from omegaconf import DictConfig

config = DictConfig({
    'motif_scaffolding': {
        'model_name': 'PPIFlow',  # 或 'RFD3'
        'motifbench_dir': '/path/to/MotifBench',
        'foldseek_database': '/path/to/foldseek/db'
    },
    'gpus': '0'
})

evaluator = Evaluation(config)

# 运行 motif scaffolding 评估
results = evaluator.run_motif_scaffolding_evaluation(
    design_dir="/path/to/design/outputs",
    pipeline_dir="/path/to/pipeline/results",
    model_name="PPIFlow",  # 可选，会使用 config 中的值
    motif_list=["01_1LDB", "02_1ITU"]  # 可选，None 表示评估所有
)

# results 是一个字典: {motif_name: result_dir}
```

### 使用方式 2: 直接使用 MotifScaffoldingEvaluation

```python
from evaluation.motif_scaffolding_evaluation import MotifScaffoldingEvaluation
from omegaconf import DictConfig

config = DictConfig({...})

# 初始化（自动选择 Generator）
motif_evaluation = MotifScaffoldingEvaluation(config, model_name='PPIFlow')

# 运行评估
results = motif_evaluation.run_motif_scaffolding_evaluation(
    design_dir="/path/to/design/outputs",
    pipeline_dir="/path/to/pipeline/results",
    motif_list=["01_1LDB", "02_1ITU"]
)
```

### Pipeline 脚本

```bash
python scripts/run_motif_scaffolding_pipeline.py \
    design_dir=/path/to/design/outputs \
    model_name=PPIFlow \
    motif_scaffolding.motif_list=[01_1LDB,02_1ITU] \
    inversefold=ProteinMPNN \
    refold=esmfold \
    gpus=0
```

### 评估流程

1. **Generator.run()**: 标准化模型输出
   - PPIFlow: 修复全ALA序列 → 真实残基名称
   - RFD3: 解析JSON → 生成 scaffold_info.csv
   - 输出: 标准化的PDB文件和 scaffold_info.csv

2. **MotifEvaluator.run_evaluation()**: 模型无关评估
   - InverseFold (ProteinMPNN): 序列设计
   - ReFold (ESMFold): 结构预测
   - Metrics: RMSD, Novelty, Diversity (alpha=5聚类)

### 输出结果

每个 motif 的评估结果保存在：
```
{pipeline_dir}/motif_scaffolding/{motif_name}/evaluation_results/
├── esm_complete_results.csv      # 完整评估结果
├── esm_novelty_results.csv       # Novelty 结果
├── summary.txt                   # 摘要（JSON格式）
└── successful_backbones/         # 成功的骨架结构
```

## 6. 其他评估任务

### Ligand Binding Protein (LBP)

```python
evaluator.run_ligand_binding_protein_evaluation(
    pipeline_dir="/path/to/pipeline",
    cands="/path/to/candidates",
    output_csv="/path/to/results.csv"
)
```

### Nucleotide Binding Ligand (NBL)

```python
evaluator.run_nuc_binding_ligand_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv"
)
```

### Protein Binding Nucleotide (PBN)

```python
evaluator.run_protein_binding_nuc_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv"
)
```

### Therapeutic Antibody/Nanobody Profile

```
# basic usage
python benchcore/scripts/run_developability_evaluation.py \
    --csv_file antibodies_fv.csv \
    --results_dir ./af3_results \
    --output developability_metrics.csv

# test single antibody
python benchcore/scripts/run_developability_evaluation.py \
    --csv_file antibodies_fv.csv \
    --results_dir ./af3_results \
    --antibody-id AB001 \
    --output test_metrics.csv
```

## 通用模式

所有评估任务都遵循相同的模式：

1. **初始化**: `evaluator = Evaluation(config)`
2. **运行评估**: `evaluator.run_<task>_evaluation(...)`
3. **获取结果**: 结果保存在指定的输出目录或CSV文件

Motif Scaffolding 的特殊之处：
- 需要指定 `model_name`（PPIFlow 或 RFD3）
- 使用 Generator 层处理模型特定的数据转换
- 评估逻辑是模型无关的
