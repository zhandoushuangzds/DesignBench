export CUDA_VISIBLE_DEVICES=6,7
# python /data/af3/qtfeng/design/odesign/benchmark/dev/remote/20251016/ODesign_benchmark/refold/esmfold/run_esmfold_distributed_2.py \
# --name test_esmfold \
# --sequences /data/af3/qtfeng/design/odesign/benchmark/dev/remote/20251016/ODesign_benchmark/examples/refold/esmfold/input/all_sequences.json \
# --output_dir /data/af3/qtfeng/design/odesign/benchmark/dev/remote/20251016/ODesign_benchmark/examples/refold/esmfold/output_3 \
# --esmfold_model_dir /data/af3/qtfeng/design/odesign/benchmark/dev/remote/20251016/ODesign_benchmark/refold/esmfold/weights \
# --verbose_gpu


torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 29500 \
  /data/af3/qtfeng/design/odesign/benchmark/dev/remote/20251020/ODesign_benchmark/refold/esmfold/run_esmfold_distributed.py \
  --name test_esmfold \
  --sequences /data/af3/qtfeng/design/odesign/benchmark/dev/remote/20251016/ODesign_benchmark/examples/refold/esmfold/input/all_sequences.json \
  --output_dir /data/af3/qtfeng/design/odesign/benchmark/dev/remote/20251016/ODesign_benchmark/examples/refold/esmfold/output_3 \
  --esmfold_model_dir /data/af3/qtfeng/design/odesign/benchmark/dev/remote/20251016/ODesign_benchmark/refold/esmfold/weights \
  --verbose_gpu

