# 你想放 wandb 的根目录（建议放到你这次实验输出目录下面）
export WANDB_ROOT=/projects/surgical-video-digital-twin/new_results/act_peg_recover/lift3d/wandb_root
mkdir -p "$WANDB_ROOT"/{runs,cache,config}

export ACT_VAE_DEBUG=1
export ACT_VAE_DEBUG_EVERY=5000

# 然后正常启动训练
# wandb 写盘位置（关键）
export WANDB_DIR="$WANDB_ROOT/runs"
export WANDB_CACHE_DIR="$WANDB_ROOT/cache"
export WANDB_CONFIG_DIR="$WANDB_ROOT/config"

# 可选：避免 wandb 再去碰 ~ 下面的 cache（有些环境会用到 XDG）
export XDG_CACHE_HOME="$WANDB_ROOT/cache"
export PYTHONPATH=$PWD/third_party:$PYTHONPATH
python -m lift3d.tools.act_policy_normal \
  --config-name=train_recover \
  benchmark=act_offline \
  agent=lift3d_act_normal \
  task_name=peg_recover \
  dataloader.batch_size=16 dataloader.num_workers=8 dataloader.shuffle=true \
  train.num_epochs=10000 train.learning_rate=1e-5 train.kl_weight=10 \
  dataset_dir=/projects/surgical-video-digital-twin/datasets/act_peg_recover/1216/zarr \
  hydra.run.dir=/projects/surgical-video-digital-twin/new_results/act_peg_recover/3dact_lr1e-6_kl10 \
  evaluation.num_skip_epochs=15 \