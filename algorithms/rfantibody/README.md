# RFantibody × DesignBench：22 靶点 × 100 设计

使用 [RFantibody](https://github.com/RosettaCommons/RFantibody)（rfdiffusion → proteinmpnn）为 DesignBench 的 22 个 antibody/nanobody 靶点各生成 100 个 design，并转换为 DesignBench 要求的 `design_dir` + `cdr_info.csv`，便于运行 `run_antibody_pipeline.py` 做评估。

## 依赖

- **RFantibody** 已安装并可调用 `rfdiffusion`、`proteinmpnn`、`qvextract`（在 RFantibody 目录下 `uv run rfdiffusion` 或激活 venv 后直接 `rfdiffusion`）
- **DesignBench** 资源完整：`assets/antibody_nanobody/antigens/*.cif`、`config/target_config.csv`、scaffolds：`antibody/hu-4D5-8_Fv.pdb`、`nanobody/h-NbBCII10.pdb`
- **CIF→PDB**：抗原为 CIF 时需 biopython，`pip install biopython`

## 一键：跑 RFantibody + 转 DesignBench 格式

在 **designbench 仓库根目录** 执行（请将 `RFantibody` 路径改为你的实际路径）：

```bash
# 若 RFantibody 在 models/RFantibody，且需用 uv 调用：
cd /path/to/RFantibody && uv run rfdiffusion --help   # 确认可用

# 在 designbench 根目录运行（指定 RFantibody 路径以便用其环境）
python algorithms/rfantibody/run_rfantibody_designbench.py \
  --designbench_root "$(pwd)" \
  --rfantibody_root /path/to/models/RFantibody \
  --output_dir ./rfantibody_designbench_output \
  --task both \
  --num_designs 100
```

若已将 RFantibody 的 venv 激活或 `rfdiffusion` 已在 PATH 中，可省略 `--rfantibody_root`：

```bash
python algorithms/rfantibody/run_rfantibody_designbench.py \
  --designbench_root "$(pwd)" \
  --output_dir ./rfantibody_designbench_output \
  --task both \
  --num_designs 100
```

- `--task both`：同时跑 antibody 与 nanobody；可改为 `antibody` 或 `nanobody`
- `--num_designs 100`：每靶点 100 个 design（DesignBench 要求每靶点最多 100）

完成后可直接用于 DesignBench 的目录为：

- **antibody**：`rfantibody_designbench_output/designbench_format/antibody/`（内含 `01_7UXQ_0.pdb` … `22_7WPC_99.pdb` 及 `cdr_info.csv`）
- **nanobody**：`rfantibody_designbench_output/designbench_format/nanobody/`（同上）

## 仅做格式转换（已有 RFantibody 输出时）

若已按靶点跑过 RFantibody，每个靶点目录下有 `rfdiffusion.qv`、`proteinmpnn.qv` 以及 `extracted/*.pdb`，只需转成 DesignBench 格式时：

```bash
python algorithms/rfantibody/run_rfantibody_designbench.py \
  --designbench_root "$(pwd)" \
  --output_dir /path/to/your_rfantibody_output \
  --convert_only \
  --task both
```

或直接调用转换脚本（nanobody 示例）：

```bash
python algorithms/rfantibody/convert_rfantibody_to_designbench.py \
  --run_dirs /path/to/output/nanobody \
  --design_dir /path/to/designbench_designs/nanobody \
  --cdr_info_csv /path/to/designbench_designs/nanobody/cdr_info.csv \
  --task nanobody \
  --max_per_target 100
```

## 用 DesignBench 跑 antibody/nanobody 评估

得到 `designbench_format/antibody` 或 `designbench_format/nanobody` 后，按 DesignBench README 跑 pipeline，例如：

```bash
# antibody (scFv/Fab)
python scripts/run_antibody_pipeline.py \
  design_dir=rfantibody_designbench_output/designbench_format/antibody \
  antibody_type=antibody \
  cdr_info_csv=rfantibody_designbench_output/designbench_format/antibody/cdr_info.csv \
  target_config_path=assets/antibody_nanobody/config/target_config.csv \
  max_designs_per_target=100 \
  gpus=0,1

# nanobody (VHH)
python scripts/run_antibody_pipeline.py \
  design_dir=rfantibody_designbench_output/designbench_format/nanobody \
  antibody_type=nanobody \
  cdr_info_csv=rfantibody_designbench_output/designbench_format/nanobody/cdr_info.csv \
  target_config_path=assets/antibody_nanobody/config/target_config.csv \
  max_designs_per_target=100 \
  gpus=0,1
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `cif_to_pdb.py` | 将抗原 CIF 转为 PDB（RFantibody 需要 PDB），依赖 biopython |
| `run_rfantibody_designbench.py` | 按靶点跑 rfdiffusion → proteinmpnn → qvextract，再调用转换脚本 |
| `convert_rfantibody_to_designbench.py` | 将每靶点的 `extracted/*.pdb` 重命名为 `{target_id}_0.pdb`…`_99.pdb`，并生成 DesignBench 要求的 `cdr_info.csv`（1-based inclusive CDR） |

## 注意事项

- 抗原在 DesignBench 中为 CIF，脚本会先转为 PDB 再交给 RFantibody；若转换失败请检查 biopython 与 CIF 是否完整。
- Part 2 靶点（12–20）在 `target_config.csv` 中 `target_hotspots` 为空，rfdiffusion 将不传 `-h`。
- 若使用 `--rfantibody_root`，脚本会在该目录下执行 rfdiffusion/proteinmpnn/qvextract，请确保该环境中已安装 RFantibody 并可用对应命令。
